import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from paretopilot.config import ConfigError, load_json, normalize_inputs, validate_model, validate_npu, validate_search
from paretopilot.agent import ExperimentAgent
from paretopilot.graph import Graph, Operator, Tensor
from paretopilot.roofline import evaluate
from paretopilot.search import pareto_front, run_search
from paretopilot.store import CacheError, RunStore


ROOT = Path(__file__).resolve().parents[1]


def tiny_model():
    return {"name": "tiny", "dtype": "fp16", "dense_features": 2, "embedding_inputs": [],
            "towers": [{"name": "ctr", "hidden_sizes": [], "output_dim": 1}]}


def tiny_npu():
    return {"name": "test", "compute_tops": {"fp16": {"matrix": 1, "vector": 1}},
            "memory_bandwidth_gbps": 1, "memory_capacity_bytes": 10**9}


def tiny_workload():
    return {"batch_size": 2, "embedding_lengths": {}}


class RooflineTests(unittest.TestCase):
    def test_hand_calculated_dense_model(self):
        result = evaluate(tiny_model(), tiny_npu(), tiny_workload())
        # 2x2 @ 2x1: 8 ops, 8 input + 4 weight + 4 output bytes.
        # Sigmoid: 4 input + 4 output bytes. At 1 GB/s total is 24 ns.
        self.assertEqual(result["metrics"]["parameter_count"], 2)
        self.assertEqual(result["metrics"]["external_bytes"], 24)
        self.assertAlmostEqual(result["metrics"]["latency_ms"], 24e-6)
        self.assertEqual(result["metrics"]["peak_activation_bytes"], 12)
        self.assertEqual(result["metrics"]["peak_memory_bytes"], 16)
        self.assertEqual(result["operators"][0]["bottleneck"], "external_memory")

    def test_compute_bound_and_padding(self):
        npu = tiny_npu()
        npu["memory_bandwidth_gbps"] = 1e6
        npu["matrix_alignment"] = 16
        result = evaluate(tiny_model(), npu, tiny_workload())
        matmul = result["operators"][0]
        self.assertEqual(matmul["executed_ops"], 2 * 16**3)
        self.assertEqual(matmul["useful_ops"], 8)
        self.assertAlmostEqual(matmul["padding_utilization"], 1 / 1024)
        self.assertEqual(matmul["bottleneck"], "compute")
        self.assertAlmostEqual(matmul["latency_ms"], 8192 / 1e12 * 1e3)

    def embedding_case(self):
        model = tiny_model()
        model["dense_features"] = 0
        model["embedding_inputs"] = [{"name": "id", "dim": 4, "pooling": "mean"}]
        workload = {"batch_size": 2, "embedding_lengths": {"id": 3}}
        return model, tiny_npu(), workload

    def test_ready_embedding_inputs_have_no_lookup_or_table_weights(self):
        model, npu, workload = self.embedding_case()
        result = evaluate(model, npu, workload)
        # Ready input [2,3,4] fp16 is 48 bytes; mean output [2,4] is 16.
        # Pool: 64 bytes, Linear: 16+8+4=28, Sigmoid: 8. No table or IDs.
        self.assertEqual(result["operators"][0]["kind"], "mean")
        self.assertEqual(result["operators"][0]["external_bytes"], 64)
        self.assertEqual(result["metrics"]["external_bytes"], 100)
        self.assertEqual(result["metrics"]["input_tensor_bytes"], 48)
        self.assertEqual(result["metrics"]["parameter_count"], 4)
        self.assertEqual(result["metrics"]["weight_bytes"], 8)
        self.assertEqual(result["metrics"]["peak_memory_bytes"], 72)
        self.assertFalse(any(op["kind"] == "embedding" for op in result["operators"]))
        self.assertFalse(any(key.endswith(".table") for key in result["graph"]["parameter_counts"]))
        self.assertTrue(result["graph"]["tensors"]["embedding_input.0"]["external"])

    def test_legacy_sfps_table_metadata_does_not_change_results(self):
        model, npu, workload = self.embedding_case()
        model["embeddings"] = model.pop("embedding_inputs")
        model["embeddings"][0]["vocab_size"] = 1000
        workload["lookups_per_sample"] = workload.pop("embedding_lengths")
        first = evaluate(model, npu, workload)
        model["embeddings"][0]["vocab_size"] = 10**12
        second = evaluate(model, npu, workload)
        self.assertEqual(first, second)
        self.assertEqual(first["model"]["schema_version"], 2)
        self.assertNotIn("vocab_size", first["model"]["embedding_inputs"][0])

    def test_legacy_sfps_cache_and_bandwidth_settings_are_ignored(self):
        model, npu, workload = self.embedding_case()
        first = evaluate(model, npu, workload)
        workload["embedding_cache_hit_rate"] = 0.99
        npu.update(random_access_efficiency=0.01, on_chip_memory_bytes=0, on_chip_bandwidth_gbps=0)
        second = evaluate(model, npu, workload)
        self.assertEqual(first, second)
        self.assertNotIn("on_chip_bytes", second["metrics"])

    def test_pooled_ready_vectors_do_not_get_pooled_twice(self):
        model, npu, workload = self.embedding_case()
        model["embedding_inputs"][0]["pooling"] = "none"
        workload["embedding_lengths"] = {}
        result = evaluate(model, npu, workload)
        self.assertEqual(result["metrics"]["input_tensor_bytes"], 16)
        self.assertEqual(result["metrics"]["external_bytes"], 36)
        self.assertEqual(result["metrics"]["peak_memory_bytes"], 28)
        self.assertEqual([o["kind"] for o in result["operators"]], ["matmul", "sigmoid"])

    def test_model_side_sum_pooling_is_still_evaluated(self):
        model, npu, workload = self.embedding_case()
        model["embedding_inputs"][0]["pooling"] = "sum"
        result = evaluate(model, npu, workload)
        self.assertEqual(result["operators"][0]["kind"], "sum")
        self.assertEqual(result["operators"][0]["useful_ops"], 2 * (3 - 1) * 4)
        self.assertEqual(result["operators"][0]["external_bytes"], 64)

    def test_input_tensor_capacity_is_not_removed_with_sfps(self):
        model, npu, workload = self.embedding_case()
        npu["memory_capacity_bytes"] = 71
        result = evaluate(model, npu, workload)
        self.assertEqual(result["status"], "infeasible")
        self.assertIn("72", result["violations"][0])

    def test_e2e_metric_and_scope_cover_every_remaining_operator(self):
        result = evaluate(*self.embedding_case())
        self.assertEqual(result["evaluation_scope"]["name"], "post_sfps_model_e2e")
        self.assertEqual(result["metrics"]["model_e2e_latency_ms"], sum(o["latency_ms"] for o in result["operators"]))
        self.assertEqual(result["metrics"]["model_e2e_latency_ms"], result["metrics"]["latency_ms"])
        self.assertNotIn("embedding", [o["kind"] for o in result["operators"]])

    def test_capacity_and_operator_support_reject(self):
        npu = tiny_npu()
        npu["memory_capacity_bytes"] = 15
        npu["supported_ops"] = ["matmul"]
        result = evaluate(tiny_model(), npu, tiny_workload())
        self.assertEqual(result["status"], "infeasible")
        self.assertEqual(len(result["violations"]), 2)

    def test_diamond_liveness_retains_shared_input(self):
        graph = Graph(tensors={
            "x": Tensor("x", (4,), 2, True), "unused": Tensor("unused", (100,), 2, True),
            "y": Tensor("y", (4,), 2), "z": Tensor("z", (4,), 2), "out": Tensor("out", (4,), 2)},
            operators=[Operator("a", "relu", ["x"], "y", 4),
                       Operator("b", "relu", ["x"], "z", 4),
                       Operator("c", "add", ["y", "z"], "out", 4)], outputs=["out"])
        self.assertEqual(graph.peak_activation_bytes(), 24)

    def test_sequence_attention_grows_quadratically(self):
        model = load_json(ROOT / "examples/sequence_model.json")
        npu = tiny_npu()
        workload = {"batch_size": 2, "embedding_lengths": {}, "sequence_length": 4}
        first = evaluate(model, npu, workload)
        workload["sequence_length"] = 8
        second = evaluate(model, npu, workload)
        a = next(o for o in first["operators"] if o["name"] == "sequence.0.attention_scores")
        b = next(o for o in second["operators"] if o["name"] == "sequence.0.attention_scores")
        self.assertEqual(b["useful_ops"], 4 * a["useful_ops"])
        self.assertEqual(b["output_shape"], [2, 4, 8, 8])

    def test_sequence_starts_with_ready_tensor_without_table(self):
        model = load_json(ROOT / "examples/sequence_model.json")
        workload = load_json(ROOT / "examples/sequence_workload.json")
        first = evaluate(model, tiny_npu(), workload)
        self.assertTrue(first["graph"]["tensors"]["sequence.input"]["external"])
        self.assertEqual(first["graph"]["tensors"]["sequence.input"]["shape"], [32, 20, 32])
        model["schema_version"] = 1
        model["sequence_encoder"]["vocab_size"] = 10**12
        second = evaluate(model, tiny_npu(), workload)
        self.assertEqual(first, second)
        self.assertFalse(any(key.endswith(".table") for key in second["graph"]["parameter_counts"]))


class ValidationTests(unittest.TestCase):
    def test_invalid_numbers_and_unknown_fields(self):
        for value in (True, 0, -1, float("nan"), "16"):
            with self.subTest(value=value):
                model = tiny_model()
                model["dense_features"] = value
                with self.assertRaises(ConfigError):
                    validate_model(model)
        npu = tiny_npu()
        npu["memory_bandwidth_gbps"] = float("inf")
        with self.assertRaises(ConfigError):
            validate_npu(npu)
        model = tiny_model()
        model["hidden_szie"] = 64
        with self.assertRaises(ConfigError):
            validate_model(model)

    def test_feature_workload_must_match(self):
        with self.assertRaises(ConfigError):
            normalize_inputs(tiny_model(), tiny_npu(), {"batch_size": 1, "embedding_lengths": {"unknown": 2}})

    def test_sequence_head_dependency(self):
        model = load_json(ROOT / "examples/sequence_model.json")
        model["sequence_encoder"]["num_heads"] = 3
        with self.assertRaises(ConfigError):
            validate_model(model)

    def test_search_cannot_change_feature_or_task_semantics(self):
        model = validate_model(tiny_model())
        for path in ("dtype", "dense_features", "towers.0.output_dim", "towers.0.name",
                     "embeddings.0.dim", "embedding_inputs.0.dim", "sequence_encoder.embedding_dim"):
            with self.subTest(path=path), self.assertRaises(ConfigError):
                validate_search({"parameters": {path: [1]}}, model)

    def test_ready_input_shapes_are_unambiguous(self):
        model = tiny_model()
        model["embedding_inputs"] = [{"name": "id", "dim": 4, "pooling": "none"}]
        with self.assertRaises(ConfigError):
            normalize_inputs(model, tiny_npu(), {"batch_size": 1, "embedding_lengths": {"id": 3}})
        model["embedding_inputs"][0]["pooling"] = "sum"
        with self.assertRaises(ConfigError):
            normalize_inputs(model, tiny_npu(), {"batch_size": 1})
        with self.assertRaises(ConfigError):
            normalize_inputs(model, tiny_npu(), {"batch_size": 1, "embedding_lengths": {"id": 1},
                                                "lookups_per_sample": {"id": 1}})

    def test_json_duplicates_and_non_finite_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            for text in ('{"a":1,"a":2}', '{"a":NaN}'):
                path.write_text(text, encoding="utf-8")
                with self.assertRaises(ConfigError):
                    load_json(path)


class SearchTests(unittest.TestCase):
    def spec(self, method="adaptive"):
        return {"method": method, "seed": 7, "max_evaluations": 8,
                "parameters": {"dense_mlp": [[], [2], [4]], "towers.0.hidden_sizes": [[], [2], [4]]},
                "constraints": {"parameter_ratio": [0.1, 100]}}

    def test_pareto_tradeoffs_and_ties(self):
        def point(cid, time, memory, status="ok"):
            return {"candidate_id": cid, "status": status,
                    "metrics": {"latency_ms": time, "peak_memory_bytes": memory}}
        points = [point("a", 1, 5), point("b", 2, 3), point("c", 3, 6),
                  point("d", 1, 5), point("e", 0.5, 1, "infeasible")]
        self.assertEqual([r["candidate_id"] for r in pareto_front(points)], ["a", "d", "b"])

    def test_adaptive_search_is_reproducible_and_bounded(self):
        a = run_search(tiny_model(), tiny_npu(), tiny_workload(), self.spec())
        b = run_search(tiny_model(), tiny_npu(), tiny_workload(), self.spec())
        self.assertEqual(a, b)
        self.assertEqual(a["summary"]["trials"], 8)
        self.assertEqual(len({t["candidate_id"] for t in a["trials"]}), 8)

    def test_grid_exhausts_without_duplicate_baseline(self):
        spec = self.spec("grid")
        spec["max_evaluations"] = 50
        result = run_search(tiny_model(), tiny_npu(), tiny_workload(), spec)
        self.assertEqual(result["summary"]["trials"], 9)
        self.assertEqual(result["summary"]["stop_reason"], "space_exhausted")

    def test_invalid_coupled_candidates_are_recorded(self):
        model = load_json(ROOT / "examples/sequence_model.json")
        workload = load_json(ROOT / "examples/sequence_workload.json")
        spec = {"method": "grid", "parameters": {"sequence_encoder.num_heads": [3, 4]}}
        result = run_search(model, tiny_npu(), workload, spec)
        self.assertEqual(result["summary"]["trials"], 2)
        self.assertEqual(result["trials"][1]["status"], "invalid")

    def test_no_feasible_result_is_not_success(self):
        npu = tiny_npu()
        npu["memory_capacity_bytes"] = 1
        result = run_search(tiny_model(), npu, tiny_workload(), self.spec())
        self.assertEqual(result["pareto"], [])
        self.assertEqual(result["summary"]["feasible"], 0)

    def test_parameter_constraint_rejects_smaller_architecture(self):
        model = tiny_model()
        model["dense_mlp"] = [4]
        spec = {"method": "grid", "parameters": {"dense_mlp": [[], [4]]},
                "constraints": {"parameter_ratio": [0.95, 1.05]}}
        result = run_search(model, tiny_npu(), tiny_workload(), spec)
        self.assertEqual(result["trials"][1]["status"], "constraint_rejected")
        self.assertEqual(len(result["pareto"]), 1)

    def test_interruption_resume_replays_same_candidates(self):
        m, n, w = normalize_inputs(tiny_model(), tiny_npu(), tiny_workload())
        spec = self.spec()
        inputs = {"model": m, "npu": n, "workload": w, "search": spec}
        with tempfile.TemporaryDirectory() as directory:
            store = RunStore(Path(directory) / "run", inputs)

            def interrupt(trial):
                if trial["trial"] == 2:
                    raise KeyboardInterrupt()

            with self.assertRaises(KeyboardInterrupt):
                run_search(m, n, w, spec, store, progress=interrupt)
            store = RunStore(store.root, inputs, resume=True)
            resumed = run_search(m, n, w, spec, store)
            fresh = run_search(m, n, w, spec)
            self.assertEqual(resumed["trials"], fresh["trials"])
            self.assertEqual(resumed["pareto"], fresh["pareto"])
            self.assertEqual(resumed["summary"]["cache_hits"], 3)

    def test_cache_integrity_and_resume_manifest(self):
        m, n, w = normalize_inputs(tiny_model(), tiny_npu(), tiny_workload())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run"
            store = RunStore(path, {"model": m, "npu": n, "workload": w})
            store.evaluate(m, n, w, evaluate)
            cached_path = next((path / "evaluations").glob("*.json"))
            cached = json.loads(cached_path.read_text(encoding="utf-8"))
            cached["result"]["metrics"]["latency_ms"] = 0
            cached_path.write_text(json.dumps(cached), encoding="utf-8")
            with self.assertRaises(ConfigError):
                store.evaluate(m, n, w, evaluate)
            with self.assertRaises(ConfigError):
                RunStore(path, {"changed": True}, resume=True)
            with self.assertRaises(ConfigError):
                RunStore(path, {"model": m})

    def test_corrupt_nonbaseline_cache_aborts_search(self):
        m, n, w = normalize_inputs(tiny_model(), tiny_npu(), tiny_workload())
        spec = self.spec("grid")
        with tempfile.TemporaryDirectory() as directory:
            store = RunStore(Path(directory) / "run", {"model": m, "npu": n, "workload": w, "search": spec})
            result = run_search(m, n, w, spec, store)
            evaluation_id = result["trials"][1]["evaluation_id"]
            (store.root / "evaluations" / (evaluation_id + ".json")).write_text("broken", encoding="utf-8")
            with self.assertRaises(CacheError):
                run_search(m, n, w, spec, store)


class CliTests(unittest.TestCase):
    def run_cli(self, *args):
        return subprocess.run([sys.executable, "-m", "paretopilot", *args], cwd=ROOT,
                              capture_output=True, text=True, encoding="utf-8")

    def test_search_exports_evaluable_candidates(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            args = ["search", "--model", "examples/model.json", "--npu", "examples/npu.json",
                    "--workload", "examples/workload.json", "--space", "examples/search.json",
                    "--output", str(output), "--quiet"]
            first = self.run_cli(*args)
            self.assertEqual(first.returncode, 0, first.stderr)
            summary = json.loads(first.stdout)
            self.assertEqual(summary["trials"], 40)
            self.assertTrue((output / "report.md").exists())
            report = (output / "report.md").read_text(encoding="utf-8")
            self.assertIn("SFPS", report)
            self.assertNotIn("累计缓存读取量", report)
            self.assertEqual(summary["evaluation_scope"]["name"], "post_sfps_model_e2e")
            exports = load_json(output / "pareto.json")
            self.assertTrue(exports)
            for item in exports:
                result = evaluate(load_json(output / item["config_path"]), load_json(ROOT / "examples/npu.json"),
                                  load_json(ROOT / "examples/workload.json"))
                self.assertEqual(result["metrics"], item["metrics"])
                self.assertEqual(item["evaluation_scope"]["name"], "post_sfps_model_e2e")
            repeat = self.run_cli(*args)
            self.assertEqual(repeat.returncode, 2)
            resumed = self.run_cli(*args, "--resume")
            self.assertEqual(resumed.returncode, 0, resumed.stderr)
            self.assertEqual(json.loads(resumed.stdout)["cache_hits"], 40)

    def test_bad_configuration_is_actionable_without_traceback(self):
        result = self.run_cli("analyze", "--model", "does-not-exist.json", "--npu", "examples/npu.json",
                              "--workload", "examples/workload.json", "--output", "runs/not-created")
        self.assertEqual(result.returncode, 2)
        self.assertIn("paretopilot:", result.stderr)
        self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()
