import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from paretopilot.backend import AscendBackend, environment
from paretopilot.build import build_candidate
from paretopilot.config import load_config
from paretopilot.util import TuningError, digest, file_hash, read_json, run_command, write_json

ROOT = Path(__file__).resolve().parents[1]


class NativeContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.c = load_config(ROOT/"examples/ascend910b_add.json")
        self.params = {k:v[0] for k,v in self.c["space"].items()}

    def test_candidate_build_isolated_and_parameters_materialized(self):
        commands = []
        def command(argv, cwd, log, timeout, env):
            commands.append(argv)
            if argv[0] == "msopgen":
                project = Path(argv[-1])
                for name in ("op_host","op_kernel","framework/onnx"):
                    (project/name).mkdir(parents=True)
                (project/"framework/onnx/parser.cpp").write_text("generated parser")
                write_json(project/"CMakePresets.json", {"configurePresets":[{"cacheVariables":{"vendor_name":{"value":"customize"}}}]})
            elif argv == ["bash","build.sh"]:
                (cwd/"build_out").mkdir()
                (cwd/"build_out/custom.run").write_text("installer")
            elif argv[0] == "bash":
                target = Path(argv[2].split("=",1)[1])/"vendors/paretopilot"
                target.mkdir(parents=True)
                (target/"kernel.so").write_bytes(b"compiled mock")
        with patch("paretopilot.build.run_command", side_effect=command):
            result = build_candidate(self.c, self.params, self.root, environment())
        self.assertTrue(Path(result["opp_path"]).is_relative_to(self.root))
        self.assertEqual(result["params_sha256"],digest(self.params))
        for folder in ("op_host","op_kernel"):
            text=(self.root/"project"/folder/"paretopilot_params.h").read_text()
            self.assertIn("#define PP_TILE_ELEMS 256",text)
            self.assertIn(digest(self.params),text)
        self.assertIn("DataCopyPad",(self.root/"project/op_kernel/pp_add.cpp").read_text())
        self.assertIn("ai.onnx::13::PpAdd",(self.root/"project/framework/onnx/parser.cpp").read_text())
        self.assertEqual(commands[0][commands[0].index("-c")+1],"ai_core-Ascend910B1")
        self.assertEqual(read_json(self.root/"source_manifest.json")["generated_source_sha256"],result["generated_source_sha256"])

    def make_backend(self):
        backend = object.__new__(AscendBackend)
        backend.c,backend.root = self.c,self.root
        backend.case = self.root/"case"
        backend.case.mkdir()
        write_json(backend.case/"case.json", {"inputs":[{"name":"a","shape":[17]},{"name":"b","shape":[17]}],"output_shape":[17]})
        backend.candidate_graph=lambda p:p.write_bytes(b"mock onnx")
        return backend

    def test_missing_tiling_signature_stops_before_runtime(self):
        backend = self.make_backend()
        called=[]
        def command(argv,directory,name,env):
            called.append(name)
            (directory/"model.om").write_bytes(b"mock")
            (directory/"atc.log").write_text("ATC done without our tiling")
        backend.command=command
        with patch("paretopilot.backend.build_candidate",return_value={"opp_path":str(self.root/"opp")}),self.assertRaisesRegex(TuningError,"parameter signature"):
            backend.evaluate(self.params,self.root,"candidate")
        self.assertEqual(called,["atc"])

    def test_changed_om_does_not_reach_profiling_export(self):
        backend=self.make_backend()
        called=[]
        def command(argv,directory,name,env):
            called.append(name)
            if name=="atc":
                (directory/"model.om").write_bytes(b"compiled")
                (directory/"atc.log").write_text("PARETOPILOT_PARAMS="+digest(self.params))
            elif name=="runtime":
                write_json(directory/"runtime.json",{"om_sha256":file_hash(directory/"model.om"),"correctness":{"passed":True}})
                (directory/"model.om").write_bytes(b"changed")
        backend.command=command
        with patch("paretopilot.backend.build_candidate",return_value={"opp_path":str(self.root/"opp")}),self.assertRaisesRegex(TuningError,"OM changed"):
            backend.evaluate(self.params,self.root,"candidate")
        self.assertEqual(called,["atc","runtime"])

    def test_environment_does_not_leak_other_opp(self):
        with patch.dict(os.environ,{"ASCEND_CUSTOM_OPP_PATH":"other/vendor"}):
            self.assertNotIn("ASCEND_CUSTOM_OPP_PATH",environment())
            self.assertEqual(environment("isolated/vendor")["ASCEND_CUSTOM_OPP_PATH"],"isolated/vendor")
            self.assertEqual(os.environ["ASCEND_CUSTOM_OPP_PATH"],"other/vendor")

    def test_process_timeout_is_logged_and_terminated(self):
        with self.assertRaises(subprocess.TimeoutExpired):
            run_command([sys.executable,"-c","import time;time.sleep(30)"],self.root,self.root/"timeout.log",0.1)
        self.assertEqual(read_json(self.root/"timeout.command.json")["timeout_s"],0.1)


if __name__ == "__main__":
    unittest.main()
