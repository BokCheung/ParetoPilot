# 配置与建模说明

输入使用 UTF-8 JSON；原生节点图还支持示例所用的静态 Jsonnet。组合式配置中的未知字段、重复键、非有限数字、非法类型和不满足依赖的维度会被拒绝。原生图适配范围见下文，示例配置均在 `examples/` 中。

## 模型配置

当前适配器描述 SFPS 之后的组合式搜推广评分网络：稠密特征经 MLP、就绪的 Embedding 张量按声明做模型内 pooling、可选的就绪行为序列经编码与 pooling，随后进行特征交互，最后进入独立任务塔。

E2E 起点是所有输入张量已在设备内存中，终点是模型输出就绪。SFPS 查表、Embedding 表参数与容量、索引读取、远程通信、输入传输与 SFPS 缓存全部在边界之外。后续模型读取输入张量的流量仍需计算。

| 字段 | 类型与含义 |
| --- | --- |
| `schema_version` | 当前为 `2`；使用 `embedding_inputs` 时可省略 |
| `name` | 模型名称 |
| `dtype` | `fp16`、`bf16`、`fp32` 或 `int8`；决定存储字节数与硬件吞吐档位 |
| `dense_features` | 每个样本的稠密输入维度，允许为 `0` |
| `embedding_inputs` | 就绪张量列表，每项含唯一 `name`、`dim`、`pooling`；无需词表规模，默认 `[]` |
| `dense_mlp` | 稠密分支各层输出宽度，默认 `[]` 表示直通 |
| `interaction` | 默认 `{"kind":"concat"}`，或 `{"kind":"dot","projection_dim":64}` |
| `sequence_encoder` | 默认 `null`，或下表中的完整序列编码配置 |
| `towers` | 至少一个任务塔，每项含唯一 `name`、`hidden_sizes` 和 `output_dim` |

上述组合式模型的 Linear 均不带 bias，MLP 隐藏层使用 ReLU，任务输出使用 Sigmoid；原生节点图的 Dense 默认包含 bias。这里的 dtype 仅描述统一的存储与运算精度档位，不实施量化或模拟数值误差。混合精度累加、量化 scale 和反量化算子尚未建模。

`pooling` 为 `mean`（默认）、`sum` 或 `none`：前两者接收 `[batch, length, dim]` 就绪张量，并计入模型侧归约的耗时与访存；`none` 接收 `[batch, dim]`，不生成额外 pooling 算子。SFPS 已返回聚合向量时应使用 `none`，避免重复计数。输入始终是浮点或指定 dtype 的 Embedding 数值，不是特征 ID。

`concat` 拼接所有分支向量。`dot` 至少需要两个分支，将各分支投影到 `projection_dim`（维度一致时直接复用），计算完整 Gram 矩阵，提取不含对角线的下三角，再拼接第一个投影向量。存在稠密分支时它是第一个向量；否则按 Embedding 列表顺序，再接序列分支。

行为序列编码配置：

| 字段 | 含义 |
| --- | --- |
| `embedding_dim` | 就绪行为序列张量的固定输入维度，不参与搜索 |
| `hidden_size` | 编码器隐藏维度 |
| `num_heads` | attention 头数，必须整除 `hidden_size` |
| `ffn_size` | FFN 中间维度 |
| `num_layers` | 编码层数 |

编码器接收 `[batch, sequence_length, embedding_dim]` 就绪输入，采用完整双向 self-attention，显式物化 attention score 矩阵；每层为 Q/K/V 投影、attention、输出投影、残差与 LayerNorm、两层 FFN、残差与 LayerNorm，最后沿序列做 mean pooling。没有自回归生成阶段，也不创建序列 Embedding 表。位置编码、mask、变长 packing、融合 attention、自定义算子和共享专家等需通过扩展 `graph.py` 适配。

## 原生 Jsonnet 节点图的范围

`examples/ads_model_config.jsonnet` 中的 `sfps_feature_embedding` 仅作为输入边界，保留节点名和输出名，供下游引用绑定；其 methods、优化器、表信息路径和通信参数不用于性能计算。后续 postprocess、split、DIN、CAN、门控、token mixer、FFN 和预测层属于模型 E2E 范围。`log_dense` 等在模型内部执行的特征变换也属于范围，不能因为出现 embedding 字样而跳过。

`validate` / `analyze` 已支持该示例的静态 Jsonnet 与 `ref::` 选择器，将 21 种模块拆解为物理算子或零成本视图，生成逐节点与 E2E Roofline。动态 Jsonnet 的 import、函数和表达式需要先在外部导出为 JSON。节点按配置顺序执行，不执行任意引用字符串中的代码。

使用 `examples/ads_workload.json` 提供输入 shape、region 和近似模型选项。该文件的分组、特征数量和序列长度是演示值；用于真实估算时必须替换。自定义业务模块的公式、配置方式及未建模部分详见 [原生逐模块说明](native-roofline.md)。原生图目前不接入结构搜索；组合式 JSON 的搜索功能保持可用。

## NPU 配置

| 字段 | 含义与默认值 |
| --- | --- |
| `name` | 硬件标识 |
| `compute_tops` | 按 dtype 提供 `matrix` 和 `vector` 两类算力，单位为每秒万亿次操作 |
| `memory_bandwidth_gbps` | 外存带宽，十进制 GB/s |
| `memory_capacity_bytes` | 设备外存容量，单位 bytes |
| `matrix_alignment` | 每个矩阵乘法 M/K/N 的计算维度对齐粒度，默认 `1` |
| `compute_efficiency` | 所有执行单元吞吐乘数，范围 `(0,1]`，默认 `1` |
| `bandwidth_efficiency` | 外存带宽乘数，范围 `(0,1]`，默认 `1` |
| `launch_overhead_us` | 每个算子的额外启动开销，默认 `0` |
| `supported_ops` | 支持的算子列表，省略时假定支持所有当前建模的算子 |

算子类型为 `matmul`、`relu`、`concat`、`gather`、`softmax`、`add`、`layernorm`、`mean`、`sum`、`sigmoid`，以及原生模块使用的 `multiply`、`subtract`、`divide`、`log`、`floor`、`dice`、`tanh`。实际硬件不支持的类型应从 `supported_ops` 中删除，不会自动假定回退到 CPU。SFPS lookup 不生成算子，也不要求设备声明支持 `embedding`。内部 `view` 仅描述张量别名，不要求设备支持，也不产生启动开销。

统一按 1 MAC = 2 operations 计数。填写浮点算力时，TFLOP/s 与这里的数值口径相同；填写整数 TOPS 时应先确认厂商的 MAC 计数口径。`matrix_alignment` 仅估算执行的额外计算量，未模拟编译器的具体 tiling 或内部 padding 存储。

## 工作负载

```json
{
  "batch_size": 128,
  "embedding_lengths": {"category": 4},
  "sequence_length": 0
}
```

- `batch_size` 是一次模型执行中的评分样本数。多个请求或多个候选需要先映射为模型实际处理的样本维度；工具不模拟在线请求调度。
- `embedding_lengths` 描述每个需要模型侧 pooling 的输入的长度，必须为正整数，且必须包含所有 `mean` / `sum` 输入。`none` 输入可省略或写 `1`，因为其输入是每样本一个就绪向量。未知特征名会被拒绝。
- 存在序列编码器时，`sequence_length` 必须为正整数；否则必须为 `0`。

## 旧配置兼容

旧版 `schema_version: 1` 的 `embeddings` 读入后转换为 `embedding_inputs`，丢弃 `vocab_size`，默认使用原来的模型侧 mean pooling。序列编码器的旧 `vocab_size` 同样丢弃。`lookups_per_sample` 作为输入长度的旧名称转换为 `embedding_lengths`，不再表示要执行查表。

旧工作负载中的 `embedding_cache_hit_rate`，以及旧 NPU 中的 `random_access_efficiency`、`on_chip_memory_bytes`、`on_chip_bandwidth_gbps` 仅做类型检查后丢弃；它们不影响评估值、规范化配置或缓存标识。旧 `supported_ops` 中的 `embedding` 也会移除。当前不提供 SRAM 分层模型。

同一配置不能同时使用新旧同义字段。新版本输入不要填写这些已移除字段。旧搜索空间中调整 `embeddings.*.dim` 或 `sequence_encoder.embedding_dim` 的规则会明确报错，因为输入形状现在固定。

## Roofline 与内存模型

对每个模型内算子，计算执行操作数 `F` 与外存搬运字节数 `D_ext`：

```text
P_eff = 对应 dtype 和执行单元的吞吐 × compute_efficiency
BW_eff = 外存带宽 × bandwidth_efficiency

t_compute = F / P_eff
t_external = D_ext / BW_eff
t_op = max(t_compute, t_external) + launch_overhead
t_model_e2e = sum(t_op)
```

矩阵乘法使用矩阵单元，其余计算使用向量单元。MatMul 按对齐后的 M/K/N 估算执行操作数，另存未 padding 的有效操作数与利用率。只搬运数据的算子可以具有零操作数，但仍占用带宽。

每个未融合算子假设从外存读取输入和模型内部权重一次、写回输出一次，不假设跨算子缓存复用。就绪输入不会产生一笔虚构的查表或 SFPS 输出写入开销；后续每个消费者对该输入的读取照常计入。`model_e2e_latency_ms` 是主延迟指标，`latency_ms` 保留为同值兼容字段。结果中的 `evaluation_scope` 明确记录起止边界。

向量运算采用简化操作计数：ReLU/残差加法每元素 1 次，Sigmoid 5 次，含缩放的 Softmax 6 次，LayerNorm 7 次；mean pooling 按输入元素数计数，sum pooling 每个输出按 `length - 1` 次加法计数。这些是解析近似，未逐项建模特殊函数吞吐、归约同步和精度转换。

峰值内存为模型内部常驻权重加上按算子顺序计算的峰值存活张量容量；含就绪输入和任务输出，物理算子使用独立输出缓冲区，最后一次使用后释放。原生 split/squeeze 视图共享源张量存储，最后一个视图释放前保留完整源分配。不包含 SFPS 表或索引，也暂不包含编译器 workspace、内存碎片和运行时额外占用。`input_tensor_bytes` 单独列出初始输入张量容量。累计搬运量不能作为内存容量。

模型假设算子内部计算与搬运理想重叠，算子之间串行，暂未建模有限 SRAM 引发的分块重复读取、DMA 竞争及启动同步。组合式模型在所有效率系数为 1 且启动开销为零时标记为**给定假设下的理想耗时下界**；否则标记为经系数调整的解析估计。原生图始终标记为近似模块估计，使用演示 workload 时还会标注 synthetic，不等价于真实实测结果。

标准 Roofline 定义可参考 [NERSC 文档](https://docs.nersc.gov/tools/performance/roofline/)。

## 搜索配置

`parameters` 是“配置路径 → 候选值列表”的映射，索引从 `0` 开始。可变字段仅限：

- `dense_mlp`，用完整宽度列表同时表达层数和各层宽度
- `interaction`，用完整对象切换 concat/dot 并保持依赖字段一致
- `towers.<index>.hidden_sizes`
- `sequence_encoder.hidden_size`、`sequence_encoder.num_heads`、`sequence_encoder.ffn_size`、`sequence_encoder.num_layers`

仅已有序列编码器支持其结构搜索。输入特征、就绪 Embedding 与序列输入维度、任务名称与输出维度等不能作为搜索字段。多个序列字段组合产生的不合法结构会记录为 `invalid`；工具不会静默修正用户的候选值。

| 字段 | 含义与默认值 |
| --- | --- |
| `method` | `adaptive`（默认）、`random` 或 `grid` |
| `seed` | 非负随机种子，默认 `0` |
| `max_evaluations` | 最大候选尝试数，含基线及拒绝候选，默认 `50` |
| `exploration_probability` | adaptive 策略随机探索概率，默认 `0.25` |
| `constraints.parameter_ratio` | 相对基线的模型内部参数量区间，排除 SFPS 表，默认 `[0.9,1.1]` |
| `constraints.max_latency_ms` | 可选的单批延迟上限 |
| `constraints.max_memory_bytes` | 可选的峰值内存上限 |

基线无论是否属于离散参数列表都会被记录，并按相同约束参与 Pareto 筛选。相同目标值的不同配置均可保留；被其他可行候选在两个目标上同时不劣、至少一个目标更优的方案会被剔除。参数量是约束，不是精度预测。

adaptive 使用当前 Pareto 集合作为优先父候选；没有可行父候选时使用已评估候选。依据父候选相关模块的耗时占比加权选择变异字段，保留随机探索，遇到重复候选时回退到未访问空间。它是有限预算的启发式搜索，不保证找到全局最优解。

## Python 调用

```python
from paretopilot.config import load_json
from paretopilot.roofline import evaluate
from paretopilot.search import run_search

model = load_json("examples/model.json")
npu = load_json("examples/npu.json")
workload = load_json("examples/workload.json")

analysis = evaluate(model, npu, workload)
search = run_search(model, npu, workload, load_json("examples/search.json"))
```

`evaluate` 会返回所有约束违反信息与解析指标，不支持的算子或内存超限使 `status` 为 `infeasible`。`run_search` 只将满足全部约束的候选放入 `pareto`。Python API 不传 `RunStore` 时不写文件。
