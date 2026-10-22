"""Keep mathematical minimum traffic distinct from profiling measurements."""
def analyze(config, duration_us=None, measured_bytes=None):
    op, hw = config["operator"], config["hardware"]
    size = 4 if op["dtype"] == "float32" else 2
    if op["kind"] == "matmul":
        m, n, k = (op[x] for x in ("m", "n", "k"))
        flops, minimum = 2*m*n*k, size*(m*k+k*n+m*n)
    else:
        flops = op["numel"] * op["flops_per_element"]
        minimum = op["numel"] * size * (op["input_count"]+1)
    traffic = minimum if measured_bytes is None else measured_bytes
    compute, memory = flops/hw["peak_flops"], traffic/hw["bandwidth_bytes_s"]
    bound = "compute" if compute >= memory else "memory"
    if op["kind"] == "matmul":
        priority = ["tile_m", "tile_n", "tile_k"] if bound == "compute" else ["tile_k", "buffers", "tile_m"]
    else:
        priority = ["tile_elems", "buffers", "block_dim"]
    return {"flops": flops, "minimum_bytes": minimum, "measured_bytes": measured_bytes,
            "traffic_basis": "shape_lower_bound" if measured_bytes is None else "profiling",
            "arithmetic_intensity": flops/traffic, "ideal_lower_bound_us": max(compute, memory)*1e6,
            "task_duration_us": duration_us, "bottleneck_hypothesis": bound, "parameter_priority": priority}
