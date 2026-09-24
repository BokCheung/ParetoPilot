"""Portable JSON, Markdown and candidate configuration exports."""

from pathlib import Path

from .agent import ExperimentAgent
from .search import diff_config
from .store import write_json


def md(value):
    return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ").replace("`", "'")


def mib(value):
    return value / (1024 * 1024)


def metrics_table(result):
    m = result["metrics"]
    return [
        "| 指标 | 估计值 |", "| --- | ---: |",
        f"| 单批推理耗时 | {m['latency_ms']:.6f} ms |",
        f"| 顺序批处理吞吐 | {m['samples_per_second']:.2f} samples/s |",
        f"| 参数量 | {m['parameter_count']:,} |",
        f"| 权重容量 | {mib(m['weight_bytes']):.4f} MiB |",
        f"| 峰值激活与输入容量 | {mib(m['peak_activation_bytes']):.4f} MiB |",
        f"| 峰值设备内存 | {mib(m['peak_memory_bytes']):.4f} MiB |",
        f"| 累计外存搬运量 | {mib(m['external_bytes']):.4f} MiB |",
        f"| 累计缓存读取量 | {mib(m['on_chip_bytes']):.4f} MiB |",
    ]


def operator_table(result):
    lines = ["| 算子 | 耗时 ms | 瓶颈 | ops/byte | padding 利用率 |", "| --- | ---: | --- | ---: | ---: |"]
    for op in sorted(result["operators"], key=lambda o: o["latency_ms"], reverse=True)[:20]:
        intensity = "n/a" if op["arithmetic_intensity"] is None else f"{op['arithmetic_intensity']:.3f}"
        lines.append(f"| {md(op['name'])} | {op['latency_ms']:.6f} | {op['bottleneck']} | {intensity} | {op['padding_utilization']:.2%} |")
    return lines


def write_analysis(directory, result):
    root = Path(directory)
    write_json(root / "analysis.json", result)
    diagnosis = ExperimentAgent().diagnose(result)
    write_json(root / "diagnosis.json", diagnosis)
    lines = [f"# ParetoPilot 分析：{md(result['model']['name'])}", "",
             f"状态：**{result['status']}**；模型质量：**未验证**。", "",
             "以下为单批模型执行的解析估计，不包含在线服务排队。", "",
             f"估计类型：`{result['estimate_kind']}`。", "", *metrics_table(result), "",
             "## 主要算子", "", *operator_table(result), "", "## 约束检查", ""]
    lines.extend(["- " + md(v) for v in result["violations"]] or ["- 通过已建模的算子支持与设备内存检查。"])
    lines.extend(["", "## 建模假设", "", *["- " + md(a) for a in result["assumptions"]], ""])
    (root / "report.md").write_text("\n".join(lines), encoding="utf-8")


def write_search(directory, result):
    root = Path(directory)
    write_json(root / "search_result.json", result)
    write_json(root / "baseline.json", result["baseline"])
    write_json(root / "diagnosis.json", result["diagnosis"])
    exports = []
    base = result["baseline"]
    for candidate in result["pareto"]:
        cid = candidate["candidate_id"]
        relative = f"candidates/{cid}.json"
        write_json(root / relative, candidate["model"])
        exports.append({"candidate_id": cid, "config_path": relative,
                        "metrics": candidate["metrics"], "quality_status": "unverified",
                        "config_diff": diff_config(base["model"], candidate["model"])})
    write_json(root / "pareto.json", exports)
    s = result["summary"]
    lines = ["# ParetoPilot 结构搜索报告", "",
             "模型质量：**未验证**。Pareto 集合基于当前 Roofline 假设与已探索候选，不代表全局最优。", "",
             f"候选尝试 {s['trials']} 个，可行 {s['feasible']} 个，Pareto 方案 {s['pareto_count']} 个。", "",
             f"停止原因：`{s['stop_reason']}`；缓存复用：{s['cache_hits']}。", "",
             "## 基线", "", *metrics_table(base), "", "## Pareto 方案", "",
             "| 配置 | 延迟 ms | 峰值内存 MiB | 参数量 | 相对基线加速比 |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for candidate in exports:
        m = candidate["metrics"]
        speedup = base["metrics"]["latency_ms"] / m["latency_ms"]
        lines.append(f"| [{candidate['candidate_id'][:12]}]({candidate['config_path']}) | {m['latency_ms']:.6f} | {mib(m['peak_memory_bytes']):.4f} | {m['parameter_count']:,} | {speedup:.3f}x |")
    if not exports:
        lines.extend(["", "没有满足全部约束的候选；检查 search_result.json 中的拒绝原因。"])
    lines.extend(["", "## Agent 分析", "", "当前 Agent 使用可复现的规则策略，不调用外部大模型。", "",
                  result["diagnosis"]["rationale"], "", "| 搜索字段 | 基线相关算子耗时占比 |", "| --- | ---: |"])
    for item in result["diagnosis"]["search_priorities"]:
        lines.append(f"| {md(item['path'])} | {item['latency_share']:.2%} |")
    lines.extend(["", "变异维度的优先级会随父候选的耗时分布更新；以上仅展示基线诊断。", "",
                  "## 基线主要算子", "", *operator_table(base), "", "## 建模假设", "",
                  *["- " + md(a) for a in base["assumptions"]], ""])
    (root / "report.md").write_text("\n".join(lines), encoding="utf-8")
