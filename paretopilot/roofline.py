"""Per-operator analytical Roofline evaluator with an explicit traffic policy."""

from __future__ import annotations

from dataclasses import asdict

from . import EVALUATOR_VERSION
from .config import normalize_inputs, fingerprint
from .graph import build_graph


ASSUMPTIONS = [
    "Model quality is unverified; no weights are loaded and no training is performed.",
    "Serial operator execution; no inter-operator overlap or online request queueing.",
    "Compute and memory transfer overlap ideally within each operator (maximum of their times).",
    "Unfused operators read inputs and weights once and write outputs once to external memory; no tile reloads.",
    "All weights stay resident in external device memory. Capacity uses tensor liveness and excludes compiler workspace.",
    "Linear layers are bias-free; hidden MLP activations are ReLU and prediction heads use sigmoid.",
    "Embedding bags use mean pooling; indices are int64. Cached rows contribute on-chip traffic.",
    "Embedding hit rate is supplied, not predicted from access distribution or cache capacity.",
    "Sequence encoder uses full bidirectional attention, materialized score matrices, post-norm blocks and mean pooling.",
    "One MAC counts as two operations. Bandwidth uses decimal GB/s; capacity is in bytes.",
    "Matrix padding affects executed work only; padded data is assumed internal, without external padding traffic.",
    "Vector operations use documented approximate operation counts; no special-function latency model.",
]


def _round_up(value, alignment):
    return (value + alignment - 1) // alignment * alignment


def evaluate(model, npu, workload):
    model, npu, workload = normalize_inputs(model, npu, workload)
    graph = build_graph(model, workload)
    input_key = {"model": model, "npu": npu, "workload": workload, "evaluator_version": EVALUATOR_VERSION}
    dtype_rates = npu["compute_tops"][model["dtype"]]
    op_results, unsupported = [], set()
    totals = {"compute": 0.0, "external_memory": 0.0, "on_chip_memory": 0.0}
    for op in graph.operators:
        if op.kind not in npu["supported_ops"]:
            unsupported.add(op.kind)
        output = graph.tensors[op.output]
        input_bytes = sum(graph.tensors[t].size_bytes for t in op.inputs)
        cache_bytes = op.lookup_bytes * workload["embedding_cache_hit_rate"]
        external_bytes = input_bytes + output.size_bytes + op.parameter_bytes_read + op.lookup_bytes - cache_bytes
        # On-chip path here covers cached embedding rows only, not all SRAM operand movement.
        executed = op.useful_ops
        if op.matrix:
            groups, m, k, n = op.matrix
            a = npu["matrix_alignment"]
            executed = 2 * groups * _round_up(m, a) * _round_up(k, a) * _round_up(n, a)
        engine = "matrix" if op.matrix else "vector"
        rate = dtype_rates[engine] * 1e12 * npu["compute_efficiency"]
        bandwidth = npu["memory_bandwidth_gbps"] * 1e9 * npu["bandwidth_efficiency"]
        if op.kind == "embedding":
            bandwidth *= npu["random_access_efficiency"]
        times = {
            "compute": executed / rate,
            "external_memory": external_bytes / bandwidth,
            "on_chip_memory": cache_bytes / (npu["on_chip_bandwidth_gbps"] * 1e9)
            if cache_bytes else 0.0,
        }
        bound = max(times, key=times.get)
        latency = times[bound] + npu["launch_overhead_us"] * 1e-6
        totals[bound] += latency
        op_results.append({
            "name": op.name, "kind": op.kind, "engine": engine,
            "input_shapes": [list(graph.tensors[t].shape) for t in op.inputs],
            "output_shape": list(output.shape),
            "useful_ops": op.useful_ops, "executed_ops": executed,
            "external_bytes": external_bytes, "on_chip_bytes": cache_bytes,
            "arithmetic_intensity": executed / external_bytes if external_bytes else None,
            "useful_arithmetic_intensity": op.useful_ops / external_bytes if external_bytes else None,
            "padding_utilization": op.useful_ops / executed if executed else 1.0,
            "effective_compute_ops_per_second": rate,
            "effective_bandwidth_bytes_per_second": bandwidth,
            "compute_time_ms": times["compute"] * 1e3,
            "external_memory_time_ms": times["external_memory"] * 1e3,
            "on_chip_memory_time_ms": times["on_chip_memory"] * 1e3,
            "latency_ms": latency * 1e3, "bottleneck": bound,
        })
    activation_bytes = graph.peak_activation_bytes()
    memory_bytes = graph.weight_bytes + activation_bytes
    violations = []
    if unsupported:
        violations.append("Unsupported operators: " + ", ".join(sorted(unsupported)))
    if memory_bytes > npu["memory_capacity_bytes"]:
        violations.append(f"Estimated memory {memory_bytes} exceeds NPU capacity {npu['memory_capacity_bytes']}")
    if workload["embedding_cache_hit_rate"]:
        largest_row = max([e["dim"] for e in model["embeddings"]] +
                          ([model["sequence_encoder"]["embedding_dim"]] if model["sequence_encoder"] else [0]))
        if largest_row * graph.element_bytes > npu["on_chip_memory_bytes"]:
            violations.append("On-chip memory cannot hold one row for the assumed embedding cache")
    latency_ms = sum(op["latency_ms"] for op in op_results)
    adjusted = any(npu[k] != 1 for k in ("compute_efficiency", "bandwidth_efficiency", "random_access_efficiency")) or npu["launch_overhead_us"] != 0
    return {
        "candidate_id": fingerprint(model), "evaluation_id": fingerprint(input_key),
        "evaluator_version": EVALUATOR_VERSION,
        "status": "infeasible" if violations else "ok", "violations": violations,
        "quality_status": "unverified",
        "estimate_kind": "adjusted_analytical_estimate" if adjusted else "ideal_roofline_lower_bound_under_assumptions",
        "model": model,
        "metrics": {
            "latency_ms": latency_ms,
            "samples_per_second": workload["batch_size"] * 1e3 / latency_ms if latency_ms else None,
            "parameter_count": graph.parameter_count, "weight_bytes": graph.weight_bytes,
            "peak_activation_bytes": activation_bytes, "peak_memory_bytes": memory_bytes,
            "useful_ops": sum(o["useful_ops"] for o in op_results),
            "executed_ops": sum(o["executed_ops"] for o in op_results),
            "external_bytes": sum(o["external_bytes"] for o in op_results),
            "on_chip_bytes": sum(o["on_chip_bytes"] for o in op_results),
            "operator_count": len(op_results),
        },
        "bottleneck_time_ms": {key: val * 1e3 for key, val in totals.items()},
        "operators": op_results,
        "graph": {"tensors": {key: {**asdict(tensor), "shape": list(tensor.shape)}
                              for key, tensor in graph.tensors.items()},
                  "operators": [{**asdict(op), "matrix": list(op.matrix) if op.matrix else None}
                                for op in graph.operators],
                  "parameter_counts": graph.parameters, "outputs": graph.outputs},
        "assumptions": ASSUMPTIONS,
    }
