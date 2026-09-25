import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from paretopilot.config import ConfigError, load_json
from paretopilot.graph import Graph, Operator, Tensor
from paretopilot.native import PROFILES, evaluate_native
from paretopilot.native_config import load_model, parse_static_jsonnet, predicate


ROOT = Path(__file__).resolve().parents[1]


def node(name, kind, inputs, parameters=None):
    return {"name": name, "type": kind, "inputs": inputs, "parameters": parameters or {}, "outputs": "output"}


def inputs_for(features, nodes, batch=2, options=None):
    model = {"format": "native_nodes", "name": "test", "nodes": [node("sfps", "sfps_feature_embedding", {}), *nodes]}
    workload = {"batch_size": batch, "dtype": "fp16", "ready_features": [
        {"name": name, "shape": shape, "region": ["test"]} for name, shape in features.items()],
        "module_options": options or {}}
    npu = {"name": "test", "compute_tops": {"fp16": {"matrix": 1, "vector": 1}},
           "memory_bandwidth_gbps": 1, "memory_capacity_bytes": 10**9}
    return model, npu, workload


class NativeParserTests(unittest.TestCase):
    def test_static_jsonnet_and_literal_strings(self):
        parsed = parse_static_jsonnet("// comment\nlocal model_structure = [{name:'a', inputs:'ref::x.y', values:[true, null, -2,],},]; model_structure")
        self.assertEqual(parsed, [{"name": "a", "inputs": "ref::x.y", "values": [True, None, -2]}])
        self.assertEqual(len(load_model(ROOT / "examples/ads_model_config.jsonnet")["nodes"]), 83)

    def test_executable_jsonnet_and_duplicates_are_rejected(self):
        for source in ("import 'x'", "local a = [1]; a + [2]", "{a: 1, a: 2}", "[1e999]", "[std.extVar('x')]", "[1] trailing"):
            with self.subTest(source=source), self.assertRaises(ConfigError):
                parse_static_jsonnet(source)

    def test_metadata_predicates_cannot_execute_code(self):
        metadata = {"name": "x", "type": "discrete", "region": ["test"]}
        self.assertTrue(predicate('type == "discrete" and name in ["x"] and "test" in region', metadata))
        for expression in ("__import__('os').getcwd()", "name.__class__", "[x for x in region]", "unknown == 1"):
            with self.subTest(expression=expression), self.assertRaises(ConfigError):
                predicate(expression, metadata)


class NativeRooflineTests(unittest.TestCase):
    def test_pooling_manual_ops_and_bytes_without_sfps(self):
        args = inputs_for({"x": [3, 4]}, [node("pool", "sum", {"inputs": "ref::sfps.output.x"}, {"axis": 1})])
        result = evaluate_native(*args)
        m = result["metrics"]
        self.assertEqual(m["useful_ops"], 2*4*(3-1))
        self.assertEqual(m["external_bytes"], 2*(2*3*4+2*4))
        self.assertAlmostEqual(m["latency_ms"], 64e-6)
        self.assertEqual(m["parameter_count"], 0)
        self.assertEqual(result["modules"][0]["status"], "excluded")
        self.assertEqual(result["modules"][0]["metrics"]["latency_ms"], 0)
        self.assertEqual(result["modules"][1]["output_shapes"], [[2, 4]])

    def test_dense_bias_activation_and_padding(self):
        args = inputs_for({"x": [3]}, [node("fc", "dense", {"inputs": "ref::sfps.output.x"}, {"units": 5, "activation": "relu"})])
        args[1]["matrix_alignment"] = 4
        result = evaluate_native(*args)
        self.assertEqual(result["metrics"]["parameter_count"], 3*5+5)
        self.assertEqual(result["metrics"]["useful_ops"], 2*2*3*5+2*5+2*5)
        self.assertEqual(result["operators"][0]["executed_ops"], 2*4*4*8)

    def test_din_manual_attention_and_sequence_scaling(self):
        din = node("din", "din_v2", {"inputs": ["ref::sfps.output.q", "ref::sfps.output.k"]},
                   {"din_specific_params": {"dnn_config": {"hidden_dims": [1], "hidden_activation": ["relu"]}}})
        args = inputs_for({"q": [1, 4], "k": [3, 4]}, [din], options={"din_v2": {"use_bias": False}})
        result = evaluate_native(*args)
        ops = {op["name"]: op for op in result["operators"]}
        # [q,k,q-k,q*k] -> [B,L,16] -> scalar; weight sequence then sum.
        self.assertEqual(ops["din.0.score.0.matmul"]["useful_ops"], 2*2*3*16)
        self.assertEqual(ops["din.0.weighted"]["useful_ops"], 2*3*4)
        self.assertEqual(ops["din.0.pool"]["useful_ops"], 2*4*(3-1))
        self.assertEqual(result["metrics"]["parameter_count"], 16)
        self.assertEqual(result["modules"][-1]["output_shapes"], [[2, 4]])
        self.assertNotIn("softmax", {op["kind"] for op in result["operators"]})
        args[2]["ready_features"][1]["shape"][0] = 6
        longer = evaluate_native(*args)
        self.assertGreater(longer["metrics"]["latency_ms"], result["metrics"]["latency_ms"])
        args[2]["module_options"]["din_v2"]["normalize_weights"] = True
        self.assertIn("softmax", {op["kind"] for op in evaluate_native(*args)["operators"]})

    def test_din_shared_weights_include_shared_dice_state(self):
        din = node("din", "din", {"inputs": "ref::sfps.output"},
                   {"din_specific_params": {"target_feature": ["q", "q"], "sequence_features": [["a"], ["b"]],
                    "dnn_config": {"hidden_dims": [3, 1], "hidden_activation": ["dice"]}}, "use_same_dnn": True})
        args = inputs_for({"q": [1, 4], "a": [3, 4], "b": [3, 4]}, [din])
        shared = evaluate_native(*args)
        args[0]["nodes"][1]["parameters"]["use_same_dnn"] = False
        unshared = evaluate_native(*args)
        self.assertEqual(unshared["metrics"]["parameter_count"], 2*shared["metrics"]["parameter_count"])
        self.assertEqual(unshared["metrics"]["useful_ops"], shared["metrics"]["useful_ops"])

    def test_can_dynamic_weights_are_inputs_not_new_model_parameters(self):
        can = node("can", "ads_can", {"inputs": [["ref::sfps.output.w"], ["ref::sfps.output.q"]]},
                   {"can_target_embedding_size": 8, "can_sequence_embedding_size": 24})
        args = inputs_for({"q": [8], "w": [24]}, [can])
        result = evaluate_native(*args)
        self.assertEqual(result["metrics"]["parameter_count"], 0)
        matrix = next(op for op in result["operators"] if op["kind"] == "matmul")
        self.assertEqual(matrix["useful_ops"], 2*2*8*3)
        self.assertEqual(matrix["external_bytes"], 2*(2*8+2*24+2*3))
        self.assertEqual(result["modules"][-1]["output_shapes"], [[2, 3]])
        args[2]["module_options"] = {"ads_can": {"hidden_sizes": [4]}}
        with self.assertRaisesRegex(ConfigError, "dynamic matrix sizes"):
            evaluate_native(*args)

    def test_per_token_ffn_weights_and_residual(self):
        ffn = node("ffn", "per_token_ffn", {"inputs": "ref::sfps.output.x"},
                   {"num_tokens": 3, "d_model": 4, "use_residual": True, "use_norm": True, "dropout": 0.5})
        args = inputs_for({"x": [3, 4]}, [ffn], options={"per_token_ffn": {"hidden_size": 8, "use_bias": False}})
        result = evaluate_native(*args)
        self.assertEqual(result["metrics"]["parameter_count"], 2*3*4*8+2*4)
        self.assertEqual(sum(op["useful_ops"] for op in result["operators"] if op["kind"] == "matmul"), 4*2*3*4*8)
        self.assertEqual(result["modules"][-1]["output_shapes"], [[2, 3, 4]])
        args[2]["module_options"]["per_token_ffn"]["shared_weights"] = True
        shared = evaluate_native(*args)
        self.assertEqual(shared["metrics"]["parameter_count"], 2*4*8+2*4)
        self.assertEqual(shared["metrics"]["useful_ops"], result["metrics"]["useful_ops"])

    def test_token_mixer_and_gate_shapes(self):
        mixer = node("mix", "token_mixer", {"inputs": "ref::sfps.output.x"}, {"num_tokens": 3, "d_model": 4})
        result = evaluate_native(*inputs_for({"x": [3, 4]}, [mixer], options={"token_mixer": {"use_bias": False}}))
        self.assertEqual(result["metrics"]["parameter_count"], 3*3)
        self.assertEqual(result["metrics"]["useful_ops"], 2*2*4*3*3)
        self.assertEqual([op["kind"] for op in result["operators"]], ["gather", "matmul", "gather"])
        gate = node("gate", "gatenu", {"inputs": ["ref::sfps.output.c", "ref::sfps.output.x"]})
        result = evaluate_native(*inputs_for({"c": [5], "x": [4]}, [gate], options={"gatenu": {"hidden_sizes": [], "use_bias": False}}))
        self.assertEqual(result["metrics"]["parameter_count"], 5*4)
        self.assertEqual(result["modules"][-1]["output_shapes"], [[2, 4]])

    def test_views_retain_parent_storage_and_do_not_launch_kernels(self):
        split = node("split", "split", {"value": "ref::sfps.output.x", "num_or_size_splits": [2, 4], "axis": -1})
        args = inputs_for({"x": [6]}, [split])
        args[1]["launch_overhead_us"] = 10
        result = evaluate_native(*args)
        self.assertEqual(result["metrics"]["latency_ms"], 0)
        self.assertEqual(result["metrics"]["external_bytes"], 0)
        self.assertEqual(result["metrics"]["peak_activation_bytes"], 24)
        # Returning a small slice must keep the whole underlying allocation alive.
        args[2]["outputs"] = ["ref::split.output[0]"]
        self.assertEqual(evaluate_native(*args)["metrics"]["peak_activation_bytes"], 24)
        graph = Graph(tensors={"parent": Tensor("parent", (100,), 2, True),
                              "slice": Tensor("slice", (10,), 2, alias_of="parent"),
                              "out": Tensor("out", (10,), 2)},
                      operators=[Operator("view", "view", ["parent"], "slice", 0),
                                 Operator("relu", "relu", ["slice"], "out", 10)], outputs=["out"])
        self.assertEqual(graph.peak_activation_bytes(), 220)

    def test_invalid_references_shapes_and_unknown_modules_fail(self):
        args = inputs_for({"x": [3]}, [node("fc", "dense", {"inputs": "ref::missing.output"}, {"units": 4})])
        with self.assertRaisesRegex(ConfigError, "reference"):
            evaluate_native(*args)
        args[0]["nodes"][1]["inputs"]["inputs"] = "ref::sfps.output.x"
        args[0]["nodes"][1]["type"] = "unknown"
        with self.assertRaisesRegex(ConfigError, "unsupported native module"):
            evaluate_native(*args)
        args = inputs_for({"x": [3]}, [node("split", "split", {"value": "ref::sfps.output.x", "num_or_size_splits": [2, 2]})])
        with self.assertRaisesRegex(ConfigError, "sum to input"):
            evaluate_native(*args)


class NativeIntegrationTests(unittest.TestCase):
    def example(self):
        return (load_model(ROOT / "examples/ads_model_config.jsonnet"), load_json(ROOT / "examples/npu.json"),
                load_json(ROOT / "examples/ads_workload.json"))

    def test_every_native_node_is_covered_and_totals_reconcile(self):
        result = evaluate_native(*self.example())
        self.assertEqual(result["status"], "ok")
        self.assertEqual(len(result["modules"]), 83)
        self.assertEqual(len({m["type"] for m in result["modules"]}), 21)
        self.assertTrue(all(m["type"] in PROFILES for m in result["modules"]))
        self.assertEqual(sum(m["status"] == "excluded" for m in result["modules"]), 1)
        for key in ("latency_ms", "useful_ops", "executed_ops", "external_bytes"):
            self.assertAlmostEqual(result["metrics"][key], sum(m["metrics"][key] for m in result["modules"]))
        self.assertEqual(result["metrics"]["parameter_count"], sum(m["parameter_count"] for m in result["modules"]))
        self.assertEqual(result["modules"][-1]["output_shapes"], [[128, 1]])
        self.assertTrue(all(not key.startswith("all_sparse_input") for key in result["graph"]["parameter_counts"]))
        changed = list(self.example())
        changed[0]["nodes"][0]["parameters"] = {"irrelevant_sfps_metadata": "anything"}
        self.assertEqual(evaluate_native(*changed)["metrics"], result["metrics"])

    def test_shape_changes_propagate_and_malformed_metadata_fails(self):
        args = list(self.example())
        base = evaluate_native(*args)
        args[2]["batch_size"] = 256
        bigger = evaluate_native(*args)
        self.assertEqual(bigger["metrics"]["parameter_count"], base["metrics"]["parameter_count"])
        self.assertEqual(bigger["metrics"]["useful_ops"], 2*base["metrics"]["useful_ops"])
        self.assertGreater(bigger["metrics"]["latency_ms"], base["metrics"]["latency_ms"])
        args[2]["dense_features"] = []
        with self.assertRaisesRegex(ConfigError, "No input features match"):
            evaluate_native(*args)

    def test_reports_cache_resume_and_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "analysis"
            command = [sys.executable, "-m", "paretopilot", "analyze", "--model", "examples/ads_model_config.jsonnet",
                       "--npu", "examples/npu.json", "--workload", "examples/ads_workload.json", "--output", str(output)]
            for extra in ([], ["--resume"]):
                process = subprocess.run(command + extra, cwd=ROOT, capture_output=True, text=True)
                self.assertEqual(process.returncode, 0, process.stderr)
                self.assertEqual(json.loads(process.stdout)["status"], "ok")
            self.assertEqual(len(list((output / "modules").glob("*.json"))), 83)
            modules = load_json(output / "modules.json")
            self.assertEqual(len(modules), 83)
            report = (output / "report.md").read_text(encoding="utf-8")
            for name in ("postprocess", "din", "can_network", "pffn2", "negative_sampling"):
                self.assertIn(name, report)
            self.assertIn("synthetic", load_json(output / "analysis.json")["estimate_kind"])

    def test_native_search_fails_before_creating_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "search"
            process = subprocess.run([sys.executable, "-m", "paretopilot", "search", "--model", "examples/ads_model_config.jsonnet",
                                      "--npu", "examples/npu.json", "--workload", "examples/ads_workload.json", "--space",
                                      "examples/search.json", "--output", str(output)], cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(process.returncode, 2)
            self.assertIn("validate/analyze", process.stderr)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
