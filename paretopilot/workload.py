"""Fixed workload shapes shared by export, measurement and Roofline."""
BASELINE_ONLY = "baseline_only"


def baseline_only(config):
    return config.get("implementation", {}).get("kind") == BASELINE_ONLY


def shapes(operator):
    if operator["kind"] == "add":
        return [[operator["numel"]], [operator["numel"]]], [operator["numel"]]
    m, n, k = (operator[key] for key in ("m", "n", "k"))
    if "batch" not in operator:
        return [[m, k], [k, n]], [m, n]
    batch = operator["batch"]
    rhs = [k, n] if operator["batch_mode"] == "shared_rhs" else [batch, k, n]
    return [[batch, m, k], rhs], [batch, m, n]
