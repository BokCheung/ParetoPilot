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
    throughput = f"{m['samples_per_second']:.2f}" if m["samples_per_second"] is not None else "n/a"
    return [
        "| 指标 | 估计值 |", "| --- | ---: |",
        f"| 模型 E2E 耗时（SFPS 之后） | {m['model_e2e_latency_ms']:.6f} ms |",
        f"| 顺序批处理吞吐 | {throughput} samples/s |",
        f"| 模型内部参数量 | {m['parameter_count']:,} |",
        f"| 模型内部权重容量 | {mib(m['weight_bytes']):.4f} MiB |",
        f"| 就绪输入张量容量 | {mib(m['input_tensor_bytes']):.4f} MiB |",
        f"| 峰值激活与输入容量 | {mib(m['peak_activation_bytes']):.4f} MiB |",
        f"| 峰值设备内存 | {mib(m['peak_memory_bytes']):.4f} MiB |",
        f"| 累计外存搬运量 | {mib(m['external_bytes']):.4f} MiB |",
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
    if "modules" in result:
        write_native_analysis(root, result)
        return
    diagnosis = ExperimentAgent().diagnose(result)
    write_json(root / "diagnosis.json", diagnosis)
    lines = [f"# ParetoPilot 分析：{md(result['model']['name'])}", "",
             f"状态：**{result['status']}**；模型质量：**未验证**。", "",
             "评估从 SFPS 输出等输入张量在设备内存就绪开始，到模型输出结束；不包含 SFPS 查表、表容量、通信、缓存和在线服务排队。", "",
             f"估计类型：`{result['estimate_kind']}`。", "", *metrics_table(result), "",
             "## 主要算子", "", *operator_table(result), "", "## 约束检查", ""]
    lines.extend(["- " + md(v) for v in result["violations"]] or ["- 通过已建模的算子支持与设备内存检查。"])
    lines.extend(["", "## 建模假设", "", *["- " + md(a) for a in result["assumptions"]], ""])
    (root / "report.md").write_text("\n".join(lines), encoding="utf-8")


def write_native_analysis(root, result):
    modules = result["modules"]
    write_json(root / "modules.json", modules)
    types = {}
    for i, module in enumerate(modules, 1):
        entry = types.setdefault(module["type"], {"type": module["type"], "count": 0, "latency_ms": 0,
                                                 "useful_ops": 0, "executed_ops": 0, "external_bytes": 0})
        entry["count"] += 1
        for key in ("latency_ms", "useful_ops", "executed_ops", "external_bytes"):
            entry[key] += module["metrics"][key]
        write_json(root / "modules" / f"{i}.json", module)
    write_json(root / "module_types.json", list(types.values()))
    lines = [f"# 模型逐模块 Roofline：{md(result['model']['name'])}", "",
             f"状态：**{result['status']}**；{len(modules)} 个节点、{len(types)} 种模块类型；模型质量未验证。", "",
             "范围：输入张量在设备内存就绪 → 最终输出。SFPS 节点仅绑定输入，查表、表容量、通信与缓存均不计入。", "",
             "**自定义模块采用近似结构，尚未用实现代码或 NPU 实测校准。** 每个节点的公式、假设、选项、输出 shape 和底层算子见下方详情与 modules.json。", ""]
    if result["workload"].get("synthetic", False):
        lines += ["**当前 workload 为演示数据**：特征分组、特征数、序列长度不代表实际业务。替换为真实 shape/region 后重新计算。", ""]
    lines += [*metrics_table(result), "", "## 计算口径", "",
              "每个物理算子 t = max(F_executed / P_engine, bytes / BW) + launch；模块耗时为其算子耗时之和，E2E 为所有模块耗时之和。不能对模块聚合后的 F/bytes 再取一次 max。", "",
              "矩阵 M/K/N 按 NPU alignment 补齐，矩阵与向量分别使用对应算力；split/squeeze/列表展开按无设备执行处理。", "",
              "## 按模块类型汇总", "", "| 类型 | 节点数 | useful ops | executed ops | 访存 bytes | 耗时 ms |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for item in sorted(types.values(), key=lambda x: x["latency_ms"], reverse=True):
        lines.append(f"| {md(item['type'])} | {item['count']} | {item['useful_ops']:,} | {item['executed_ops']:,} | {item['external_bytes']:,} | {item['latency_ms']:.6f} |")
    lines += ["", "## 每个节点", "", "| 节点 | 类型 | useful ops | 访存 bytes | ops/byte（含 padding） | 耗时 ms | 瓶颈 |",
              "| --- | --- | ---: | ---: | ---: | ---: | --- |"]
    for module in modules:
        m = module["metrics"]
        intensity = "—" if m["arithmetic_intensity"] is None else f"{m['arithmetic_intensity']:.3f}"
        lines.append(f"| {md(module['name'])} | {md(module['type'])} | {m['useful_ops']:,} | {m['external_bytes']:,} | {intensity} | {m['latency_ms']:.6f} | {md(module['bottleneck'])} |")
    lines += ["", "## 节点计算详情", ""]
    by_name = {op["name"]: op for op in result["operators"]}
    for i, module in enumerate(modules, 1):
        # Numeric filenames avoid trusting user-supplied node names as paths.
        lines += [f"### {md(module['name'])} / {md(module['type'])}", "",
                  f"状态：{module['status']}；置信类别：{module['confidence']}。", "",
                  f"公式：`{md(module['formula'])}`。", "", f"假设：{module['assumption']}", "",
                  f"选项覆盖：`{md(module['options'])}`；输出 shape：`{md(module['output_shapes'])}`。", "",
                  f"模型内部参数/状态量：{module['parameter_count']:,}；[节点 JSON](modules/{i}.json)。", "",
                  "| 算子 | 引擎 | useful ops | executed ops | bytes | compute ms | memory ms | latency ms |",
                  "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for name in module["operator_names"]:
            op = by_name[name]
            lines.append(f"| {md(name)} | {op['engine']} | {op['useful_ops']:,} | {op['executed_ops']:,} | {op['external_bytes']:,} | {op['compute_time_ms']:.6f} | {op['external_memory_time_ms']:.6f} | {op['latency_ms']:.6f} |")
        if not module["operator_names"]:
            lines += ["| 无设备算子 | — | 0 | 0 | 0 | 0 | 0 | 0 |"]
        lines += [""]
    lines += ["## 约束与假设", "", *["- " + md(v) for v in result["violations"]],
              *["- " + md(a) for a in result["assumptions"]], ""]
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
                        "evaluation_scope": candidate["evaluation_scope"],
                        "metrics": candidate["metrics"], "quality_status": "unverified",
                        "config_diff": diff_config(base["model"], candidate["model"])})
    write_json(root / "pareto.json", exports)
    s = result["summary"]
    lines = ["# ParetoPilot 结构搜索报告", "",
             "评估边界：输入张量在设备内存就绪 → 模型输出；SFPS 查表、表容量、通信与缓存不计入结果。", "",
             "模型质量：**未验证**。Pareto 集合基于当前 Roofline 假设与已探索候选，不代表全局最优。", "",
             f"候选尝试 {s['trials']} 个，可行 {s['feasible']} 个，Pareto 方案 {s['pareto_count']} 个。", "",
             f"停止原因：`{s['stop_reason']}`；缓存复用：{s['cache_hits']}。", "",
             "## 基线", "", *metrics_table(base), "", "## Pareto 方案", "",
             "| 配置 | 模型 E2E 延迟 ms | 峰值内存 MiB | 模型内部参数量 | 相对基线加速比 |",
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
