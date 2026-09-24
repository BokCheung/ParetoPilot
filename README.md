# ParetoPilot

面向搜索、推荐与广告（搜推广）场景的 NPU 硬件感知模型结构自动优化 Agent。

ParetoPilot 面向搜推广模型的推理优化，旨在以模型配置、NPU 硬件规格和推理工作负载为输入，通过 Roofline 性能建模与自动化搜索，在给定模型规模和结构约束下，发现更适合目标硬件的模型配置。

系统围绕 Embedding 维度、MLP 层数与各层宽度、特征交互模块、行为序列编码模块和多任务塔结构等参数构建设计空间，具体可调整项由模型类型决定。系统将候选模型转换为算子图，分析 Embedding lookup、特征交互、矩阵乘法、序列编码等算子的计算量、数据搬运量与算术强度，再结合 NPU 的算力、内存带宽和存储容量，估算推理延迟与内存占用，识别计算或带宽瓶颈。

## 核心能力

当前版本提供以下能力：

- **模型结构解析与约束管理**：建立配置字段、张量维度和算子之间的依赖关系，生成合法候选结构。
- **Roofline 性能评估**：在指定 batch 和特征负载下，分析模型单批推理的逐算子耗时、瓶颈分布和整网性能，覆盖稀疏特征访问与稠密计算。
- **多目标自动搜索**：在参数量范围和硬件约束下，搜索延迟与内存占用之间的 Pareto 方案。
- **Agent 实验规划与解释**：根据评估结果调整搜索方向，解释结构变化的预期收益及其依据。
- **可复现实验记录**：保存候选配置、配置差异、工作负载、建模假设和评估结果。

## 输入与输出

| 类型 | 内容 |
| --- | --- |
| 模型配置 | 组合式模型结构、Embedding 表规模与维度、MLP 层数与宽度、特征交互、行为序列编码、多任务塔配置及结构依赖 |
| NPU 信息 | 各精度与执行单元的算力、内存带宽、存储容量、支持的算子及 shape 对齐要求 |
| 推理工作负载 | batch size（每批评分样本数）、每样本 Embedding lookup 数量、行为序列长度及假定的缓存命中率；特征规模由模型配置提供 |
| 搜索约束 | 可调整字段、取值范围、模型规模限制、硬件约束及搜索预算 |
| 输出结果 | 候选模型配置、相对基线的配置差异、性能预测、瓶颈分析和 Pareto 集合 |

## 工作流程

```text
模型配置 + NPU 信息 + 推理负载
    → 结构搜索与合法性检查
    → 算子图及计算量、访存量分析
    → Roofline 性能评估
    → Pareto 筛选与 Agent 分析
    → 候选模型配置与优化报告
```

搜索器根据评估反馈迭代生成候选；Agent 根据瓶颈分析提出搜索方向，并组织实验和解释结果。

## 第一阶段范围

第一阶段聚焦固定 NPU 上搜推广模型的结构性能探索，固定输入特征定义和任务输出语义，暂不包含微调、蒸馏和模型质量恢复。通过限制模型规模和结构范围，控制搜索边界，避免仅以缩小模型获得性能收益；这些限制不构成模型质量保证。

Roofline 以算子或明确的融合算子组为评估单位。采用峰值算力与带宽时，得到的是给定计算量和访存假设下的理想耗时下界；后续可通过实测数据校准。峰值内存占用单独计算，不与累计数据搬运量混用。

Embedding 表的存储容量与每批 lookup 的数据搬运量分别建模，并记录缓存命中、访问分布与有效带宽假设。性能指标首先描述给定工作负载下的模型执行，不包含在线服务的排队和请求调度时间。

输出方案提供基于明确假设的性能预测，模型质量标记为未验证，为后续训练和部署验证提供候选结构与分析依据。

## 快速开始

需要 Python 3.10 或更新版本。核心功能仅使用 Python 标准库，在仓库根目录即可运行，无需安装第三方依赖。

校验配置与算子图：

```bash
python -m paretopilot validate --model examples/model.json --npu examples/npu.json --workload examples/workload.json --space examples/search.json
```

分析基线模型：

```bash
python -m paretopilot analyze --model examples/model.json --npu examples/npu.json --workload examples/workload.json --output runs/baseline
```

运行结构搜索：

```bash
python -m paretopilot search --model examples/model.json --npu examples/npu.json --workload examples/workload.json --space examples/search.json --output runs/search
```

行为序列模型示例：

```bash
python -m paretopilot search --model examples/sequence_model.json --npu examples/npu.json --workload examples/sequence_workload.json --space examples/sequence_search.json --output runs/sequence
```

`examples/npu.json` 是演示用的合成硬件规格，不对应真实产品。使用前请替换为目标 NPU 数据。当前模型配置采用项目定义的组合式格式，不直接读取任意框架的原生模型配置；字段、计算规则及适配范围见 [配置与建模说明](docs/configuration.md)。

## 搜索与 Agent

支持 `grid`、`random` 和 `adaptive` 三种策略，默认使用 `adaptive`。基线始终作为第一条实验记录；随后在离散设计空间中生成不重复的候选。

当前 Agent 是可复现的规则策略：读取父候选的逐算子耗时，对主要瓶颈所在模块的可变字段提高采样权重，同时保留随机探索。它无需外部大模型服务，不把参数量视为模型质量。每次变异的来源、字段和决策依据保存在实验记录中。

默认限制参数量为基线的 90%–110%，可以通过 `constraints.parameter_ratio` 调整。Embedding 表词表规模、输入特征定义、精度和任务输出维度保持固定；结构变换会自动重建下游张量形状。

搜索预算 `max_evaluations` 包含基线和被结构校验或约束拒绝的候选。Pareto 集合最小化单批延迟与峰值内存，只收录满足模型、硬件和搜索约束的已评估方案。

## 实验产物与恢复

输出目录默认必须为空，避免覆盖已有实验。主要文件包括：

| 文件 | 内容 |
| --- | --- |
| `manifest.json` | 规范化输入、搜索设置、版本及运行标识 |
| `analysis.json` | 单点评估结果、算子图、逐算子指标与假设 |
| `search_result.json` | 搜索结果、全部候选尝试及拒绝原因 |
| `baseline.json` | 搜索基线的完整评估结果 |
| `pareto.json` | 可行 Pareto 集合及相对基线的配置差异 |
| `candidates/` | 可再次输入本工具的候选模型配置 |
| `diagnosis.json` | Agent 的瓶颈诊断与搜索优先级 |
| `report.md` | 可阅读的中文分析或搜索报告 |
| `evaluations/` | 带校验和的逐候选评估缓存 |
| `progress.json` | 已完成的搜索尝试 |

中断后使用相同命令并追加 `--resume`。系统检查输入与版本一致性，按相同随机种子重放搜索决策并复用已完成的评估。修改模型、硬件、工作负载、搜索预算或其他搜索设置时，请使用新的输出目录。

命令成功返回 `0`；配置错误、硬件不可行或搜索无可行方案返回 `2`；用户中断返回 `130`。`validate` 仅检查配置和算子图，硬件可行性由 `analyze` 或 `search` 检查。

## 开发与验证

```bash
python -m unittest discover -s tests -v
```

测试覆盖手算 Roofline、Embedding 容量与访存分离、矩阵 padding、分支张量生命周期、序列 attention、约束与 Pareto 筛选、缓存完整性、中断恢复及命令行导出。

代码按 `config`、`graph`、`roofline`、`search`、`agent`、`store`、`report` 和 `cli` 划分，可独立扩展新的模型适配器或评估后端。
