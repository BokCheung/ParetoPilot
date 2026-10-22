def analyze(config, duration_us=None, measured_bytes=None):
    op, hw = config["operator"], config["hardware"]
    size = 2 if op["dtype"] == "float16" else 4
    if op["kind"] == "matmul":
        m, n, k = op["m"], op["n"], op["k"]
        flops, minimum = 2*m*n*k, size*(m*k+k*n+m*n)
        priority = ["tile_m", "tile_n", "tile_k"]
    else:
        flops, minimum = op["numel"], 3*size*op["numel"]
        priority = ["tile_elems", "buffers", "block_dim"]
    traffic = minimum if measured_bytes is None else measured_bytes
    compute, memory = flops/hw["peak_flops"], traffic/hw["bandwidth_bytes_s"]
    if op["kind"] == "matmul" and memory > compute:
        priority = ["tile_k", "buffers", "tile_m"]
    return {"flops": flops, "minimum_bytes": minimum, "measured_bytes": measured_bytes,
            "traffic_basis": "shape_minimum" if measured_bytes is None else "profiling",
            "arithmetic_intensity": flops/traffic, "ideal_lower_bound_us": max(compute, memory)*1e6,
            "task_duration_us": duration_us, "bottleneck_hypothesis": "compute" if compute >= memory else "memory",
            "suggested_priority": priority}
