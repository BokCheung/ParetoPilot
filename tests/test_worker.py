import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from paretopilot.config import load_config
from paretopilot.util import TuningError, read_json
from paretopilot.worker import check_outputs, execute_om, export_case

ROOT = Path(__file__).resolve().parents[1]
HAS_NUMPY = importlib.util.find_spec("numpy") is not None
HAS_EXPORT = all(importlib.util.find_spec(m) is not None for m in ("numpy","torch","onnx"))


class FakeACL:
    """CPU byte-buffer emulator to exercise runner ordering and cleanup, not NPU math."""
    def __init__(self, arrays, reference, wrong=False, fail_execute=False):
        self.events, self.memory, self.nextptr = [], {}, 100
        self.profiled = False
        self.arrays, self.reference = arrays, reference
        def call(name, result=0):
            def function(*args):
                self.events.append(name)
                return result
            return function
        def allocate(size, *args):
            self.nextptr += 1
            self.memory[self.nextptr] = bytes(size)
            return self.nextptr,0
        def free(ptr):
            del self.memory[ptr]
            return 0
        def memcpy(dst, capacity, src, size, kind):
            raw = src if isinstance(src,bytes) else self.memory[src]
            self.memory[dst] = raw[:size]
            return 0
        self.init, self.finalize = call("init"), call("finalize")
        self.create_data_buffer = lambda ptr,size:(ptr,size)
        self.destroy_data_buffer = call("destroy_buffer")
        self.rt = SimpleNamespace(set_device=call("set_device"),reset_device=call("reset_device"),create_context=call("context",(1,0)),
            destroy_context=call("destroy_context"),get_run_mode=lambda:(1,0),malloc=allocate,free=free,malloc_host=allocate,free_host=free,memcpy=memcpy)
        self.util = SimpleNamespace(bytes_to_ptr=lambda b:b,ptr_to_bytes=lambda p,n:self.memory[p][:n])
        def buffer(ds, buf):
            ds.append(buf)
            return ds,0
        def execute(model, inputs, outputs):
            self.events.append("sample" if self.profiled else "unprofiled")
            if fail_execute:
                return 123
            out = self.reference+1 if wrong else self.reference
            ptr,size = outputs[0]
            self.memory[ptr] = out.tobytes()+bytes(size-out.nbytes)
            return 0
        self.mdl = SimpleNamespace(load_from_file=call("load",(1,0)),unload=call("unload"),create_desc=lambda:1,destroy_desc=call("destroy_desc"),
            get_desc=call("get_desc"),get_num_inputs=lambda d:2,get_num_outputs=lambda d:1,create_dataset=lambda:[],destroy_dataset=call("destroy_dataset"),
            add_dataset_buffer=buffer,execute=execute)
        for direction,values in (("input",arrays),("output",[reference])):
            setattr(self.mdl, "get_"+direction+"_dims", lambda d,i,values=values:({"dims":list(values[i].shape)},0))
            setattr(self.mdl, "get_"+direction+"_data_type", lambda d,i:1)
            setattr(self.mdl, "get_"+direction+"_format", lambda d,i:2)
            setattr(self.mdl, "get_"+direction+"_size_by_index", lambda d,i,values=values:((values[i].nbytes+31)//32)*32)
        def start(*args):
            self.events.append("start")
            self.profiled = True
            return 0
        def stop(*args):
            self.events.append("stop")
            self.profiled = False
            return 0
        self.prof=SimpleNamespace(init=call("prof_init"),finalize=call("prof_finalize"),create_config=lambda *args:1,
            destroy_config=call("destroy_config"),start=start,stop=stop)


@unittest.skipUnless(HAS_NUMPY,"numpy not installed")
class WorkerTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.c = load_config(ROOT/"examples/ascend910b_add.json")
        self.c["measurement"].update(warmup=3,repeats=4)
        self.arrays = [np.arange(17,dtype=np.float16), np.ones(17,dtype=np.float16)]
        self.ref = self.arrays[0]+self.arrays[1]
        for i,array in enumerate(self.arrays):
            np.save(self.root/f"input_{i}.npy",array)
        np.save(self.root/"reference.npy",self.ref)
        (self.root/"model.om").write_bytes(b"mock model")
        self.request = {"config":self.c,"directory":str(self.root),"case":str(self.root),"om":str(self.root/"model.om")}

    def test_correctness_rejects_nonfinite_wrong_dtype_and_value(self):
        import numpy as np
        self.assertTrue(check_outputs(self.ref,self.ref,self.c["measurement"])["passed"])
        self.assertFalse(check_outputs(self.ref+2,self.ref,self.c["measurement"])["passed"])
        self.assertFalse(check_outputs(np.full_like(self.ref,np.nan),self.ref,self.c["measurement"])["passed"])
        with self.assertRaises(TuningError):
            check_outputs(self.ref.astype(np.float32),self.ref,self.c["measurement"])

    def test_warmup_excluded_and_padded_nd_buffers_supported(self):
        acl = FakeACL(self.arrays,self.ref)
        with patch.dict(sys.modules, {"acl":acl}):
            execute_om(self.request)
        self.assertEqual(acl.events.count("sample"),4)
        self.assertEqual(acl.events.count("unprofiled"),4)
        self.assertLess(acl.events.index("load"),acl.events.index("start"))
        self.assertEqual(acl.memory,{})
        self.assertEqual(acl.events[-1],"finalize")
        self.assertTrue(read_json(self.root/"runtime.json")["correctness"]["passed"])

    def test_bad_correctness_prevents_profiling_and_cleans_up(self):
        acl = FakeACL(self.arrays,self.ref,wrong=True)
        with patch.dict(sys.modules,{"acl":acl}),self.assertRaises(TuningError):
            execute_om(self.request)
        self.assertNotIn("start",acl.events)
        self.assertFalse(read_json(self.root/"correctness.json")["passed"])
        self.assertFalse((self.root/"runtime.json").exists())
        self.assertEqual(acl.memory,{})

    def test_acl_error_cleans_up(self):
        acl = FakeACL(self.arrays,self.ref,fail_execute=True)
        with patch.dict(sys.modules,{"acl":acl}),self.assertRaises(TuningError):
            execute_om(self.request)
        self.assertEqual(acl.memory,{})
        self.assertIn("unload",acl.events)


@unittest.skipUnless(HAS_EXPORT,"optional torch/onnx/numpy not installed")
class ExportTests(unittest.TestCase):
    def test_real_pytorch_export_for_builtin_and_python_baseline(self):
        import onnx
        import numpy as np
        with tempfile.TemporaryDirectory() as temp:
            for kind in ("matmul","add","python"):
                target = Path(temp)/kind
                target.mkdir()
                c = load_config(ROOT/("examples/ascend910b_add.json" if kind=="add" else "examples/ascend910b_matmul.json"))
                if kind=="python":
                    c["baseline"]={"kind":"python","path":str(ROOT/"examples/baseline_matmul.py")}
                export_case(c,target)
                model = onnx.load(str(target/"baseline.onnx"))
                self.assertEqual([n.op_type for n in model.graph.node],["Add" if kind=="add" else "MatMul"])
                self.assertTrue(np.isfinite(np.load(target/"reference.npy")).all())
                self.assertEqual(model.graph.input[0].name,"a")


if __name__ == "__main__":
    unittest.main()
