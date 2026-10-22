"""Heavy dependencies and ACL device state live in a short-lived subprocess."""
from contextlib import ExitStack
import importlib.util
import inspect
from pathlib import Path
import sys
from .util import TuningError, file_hash, read_json, write_json


def export_case(config, directory):
    import numpy as np
    import torch
    import onnx
    directory = Path(directory)
    op = config["operator"]
    dtype = getattr(torch, op["dtype"])
    shapes = [[op["m"], op["k"]], [op["k"], op["n"]]] if op["kind"] == "matmul" else [[op["numel"]]]*2
    expected_output = [op["m"], op["n"]] if op["kind"] == "matmul" else [op["numel"]]
    if config["baseline"]["kind"] == "python":
        spec = importlib.util.spec_from_file_location("user_baseline", config["baseline"]["path"])
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        model, inputs = module.build_case(op, config["search"]["seed"])
    else:
        class Baseline(torch.nn.Module):
            def forward(self, a, b):
                return torch.matmul(a, b) if op["kind"] == "matmul" else a+b
        model = Baseline()
        rng = torch.Generator().manual_seed(config["search"]["seed"])
        inputs = tuple(torch.randn(shape, generator=rng, dtype=dtype) for shape in shapes)
    model = model.cpu().eval()
    inputs = tuple(x.detach().cpu().contiguous() for x in inputs)
    if len(inputs) != 2 or any(list(x.shape) != shape or x.dtype != dtype for x, shape in zip(inputs, shapes)):
        raise TuningError("Baseline input tensors differ from fixed operator description")
    with torch.no_grad():
        output = model(*inputs)
    if not isinstance(output, torch.Tensor) or list(output.shape) != expected_output or output.dtype != dtype:
        raise TuningError("Baseline output must have the declared fixed shape/dtype")
    for i, value in enumerate(inputs):
        np.save(directory/f"input_{i}.npy", value.numpy())
    np.save(directory/"reference.npy", output.detach().numpy())
    exporter_options = {"dynamo": False} if "dynamo" in inspect.signature(torch.onnx.export).parameters else {}
    torch.onnx.export(model, inputs, str(directory/"baseline.onnx"), input_names=["a", "b"], output_names=["c"],
                      opset_version=13, do_constant_folding=False, **exporter_options)
    onnx.checker.check_model(onnx.load(str(directory/"baseline.onnx")))
    write_json(directory/"case.json", {"inputs": [{"name": n, "shape": shape, "dtype": op["dtype"]} for n, shape in zip(["a", "b"], shapes)],
                                       "output_shape": expected_output, "dtype": op["dtype"], "seed": config["search"]["seed"]})


def check_outputs(actual, reference, measurement):
    import numpy as np
    if actual.shape != reference.shape or actual.dtype != reference.dtype:
        raise TuningError(f"Output shape/dtype mismatch: {actual.shape}/{actual.dtype}, expected {reference.shape}/{reference.dtype}")
    a, b = actual.astype(np.float64), reference.astype(np.float64)
    finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
    error = np.abs(a-b)
    result = {"passed": finite and bool(np.allclose(a, b, rtol=measurement["rtol"], atol=measurement["atol"], equal_nan=False)),
              "max_abs_error": float(error.max()) if finite else None,
              "max_relative_error": float((error/np.maximum(np.abs(b), 1e-12)).max()) if finite else None,
              "atol": measurement["atol"], "rtol": measurement["rtol"], "reference": "pytorch"}
    return result


def execute_om(request):
    import acl
    import numpy as np
    directory, case = Path(request["directory"]), Path(request["case"])
    config = request["config"]
    m, device = config["measurement"], config["ascend"]["device"]
    inputs = [np.load(case/f"input_{i}.npy", allow_pickle=False) for i in range(2)]
    reference = np.load(case/"reference.npy", allow_pickle=False)
    profiler = directory/"profiler"
    profiler.mkdir()
    def ok(code, label):
        if code != 0:
            raise TuningError(f"{label} failed with ACL code {code}")
    dtype_map = {0: np.dtype('float32'), 1: np.dtype('float16')}
    with ExitStack() as cleanup:
        ok(acl.init(), "acl.init")
        cleanup.callback(acl.finalize)
        ok(acl.rt.set_device(device), "set_device")
        cleanup.callback(acl.rt.reset_device, device)
        context, ret = acl.rt.create_context(device)
        ok(ret, "create_context")
        cleanup.callback(acl.rt.destroy_context, context)
        run_mode, ret = acl.rt.get_run_mode()
        ok(ret, "get_run_mode")
        if run_mode != 1:
            raise TuningError("Built-in runtime requires ACL_HOST mode")
        # Initialize before model load to collect metadata, start after warmup only.
        ok(acl.prof.init(str(profiler)), "prof.init")
        cleanup.callback(acl.prof.finalize)
        model, ret = acl.mdl.load_from_file(request["om"])
        ok(ret, "load_from_file")
        cleanup.callback(acl.mdl.unload, model)
        desc = acl.mdl.create_desc()
        cleanup.callback(acl.mdl.destroy_desc, desc)
        ok(acl.mdl.get_desc(desc, model), "get_desc")
        if acl.mdl.get_num_inputs(desc) != 2 or acl.mdl.get_num_outputs(desc) != 1:
            raise TuningError("OM must have exactly two inputs and one output")
        datasets, output_ptr, output_size = [], None, None
        for direction, arrays in (("input", inputs), ("output", [reference])):
            dataset = acl.mdl.create_dataset()
            cleanup.callback(acl.mdl.destroy_dataset, dataset)
            for i, array in enumerate(arrays):
                dims, ret = getattr(acl.mdl, f"get_{direction}_dims")(desc, i)
                ok(ret, "get_dims")
                dt = getattr(acl.mdl, f"get_{direction}_data_type")(desc, i)
                size = getattr(acl.mdl, f"get_{direction}_size_by_index")(desc, i)
                layout = getattr(acl.mdl, f"get_{direction}_format")(desc, i)
                if list(array.shape) != dims["dims"] or dtype_map.get(dt) != array.dtype or size < array.nbytes or layout != 2:
                    raise TuningError(f"OM {direction} {i}: shape/dtype/size mismatch or non-ND external layout")
                ptr, ret = acl.rt.malloc(size, 0)
                ok(ret, "malloc")
                cleanup.callback(acl.rt.free, ptr)
                buffer = acl.create_data_buffer(ptr, size)
                cleanup.callback(acl.destroy_data_buffer, buffer)
                _, ret = acl.mdl.add_dataset_buffer(dataset, buffer)
                ok(ret, "add_dataset_buffer")
                if direction == "input":
                    raw = array.tobytes() + bytes(size-array.nbytes)
                    ok(acl.rt.memcpy(ptr, size, acl.util.bytes_to_ptr(raw), size, 1), "H2D")
                else:
                    output_ptr, output_size = ptr, size
            datasets.append(dataset)
        def execute():
            ok(acl.mdl.execute(model, datasets[0], datasets[1]), "mdl.execute")
        def compare():
            host, ret = acl.rt.malloc_host(output_size)
            ok(ret, "malloc_host")
            try:
                ok(acl.rt.memcpy(host, output_size, output_ptr, output_size, 2), "D2H")
                raw = acl.util.ptr_to_bytes(host, output_size)
                actual = np.frombuffer(raw[:reference.nbytes], dtype=reference.dtype).reshape(reference.shape).copy()
                np.save(directory/"output.npy", actual)
                result = check_outputs(actual, reference, m)
                write_json(directory/"correctness.json", result)
                if not result["passed"]:
                    raise TuningError("Candidate output failed PyTorch correctness comparison")
                return result
            finally:
                acl.rt.free_host(host)
        execute()
        compare()
        for _ in range(m["warmup"]):
            execute()
        prof_config = acl.prof.create_config([device], config["ascend"].get("aicore_metrics", 0), 0, 0x1 | 0x2 | 0x4)
        if not prof_config:
            raise TuningError("prof.create_config returned null")
        cleanup.callback(acl.prof.destroy_config, prof_config)
        active = [False]
        def stop():
            if active[0]:
                ok(acl.prof.stop(prof_config), "prof.stop")
                active[0] = False
        cleanup.callback(stop)
        ok(acl.prof.start(prof_config), "prof.start")
        active[0] = True
        for _ in range(m["repeats"]):
            execute()
        stop()
        correctness = compare()
    write_json(directory/"runtime.json", {"status": "ok", "om_sha256": file_hash(request["om"]), "correctness": correctness,
                                           "warmup_excluded": True, "warmup": m["warmup"], "profiled_iterations": m["repeats"]})


def main():
    request = read_json(sys.argv[2])
    if sys.argv[1] == "export":
        export_case(request["config"], request["directory"])
    elif sys.argv[1] == "run":
        execute_om(request)
    else:
        raise TuningError("Unknown worker action")


if __name__ == "__main__":
    main()
