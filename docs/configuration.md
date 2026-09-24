# 配置与建模说明

所有输入使用 UTF-8 JSON。未知字段、重复 JSON 键、非有限数字、非法类型和不满足依赖的维度会被拒绝。示例配置均在 `examples/` 中。

## 模型配置

当前适配器描述一个组合式搜推广评分网络：稠密特征经 MLP、稀疏特征经 Embedding lookup 和 mean pooling、可选行为序列经编码与 pooling，随后进行特征交互，最后进入独立任务塔。

| 字段 | 类型与含义 |
| --- | --- |
| `schema_version` | 当前为 `1`，可省略 |
| `name` | 模型名称 |
| `dtype` | `fp16`、`bf16`、`fp32` 或 `int8`；决定存储字节数与硬件吞吐档位 |
| `dense_features` | 每个样本的稠密输入维度，允许为 `0` |
| `embeddings` | 特征表列表，每项含唯一 `name`、`vocab_size`、`dim` |
| `dense_mlp` | 稠密分支各层输出宽度，默认 `[]` 表示直通 |
| `interaction` | 默认 `{"kind":"concat"}`，或 `{"kind":"dot","projection_dim":64}` |
| `sequence_encoder` | 默认 `null`，或下表中的完整序列编码配置 |
| `towers` | 至少一个任务塔，每项含唯一 `name`、`hidden_sizes` 和 `output_dim` |

所有 Linear 均不带 bias，MLP 隐藏层使用 ReLU，任务输出使用 Sigmoid。这里的 dtype 仅描述统一的存储与运算精度档位，不实施量化或模拟数值误差。混合精度累加、量化 scale 和反量化算子尚未建模。

`concat` 拼接所有分支向量。`dot` 至少需要两个分支，将各分支投影到 `projection_dim`（维度一致时直接复用），计算完整 Gram 矩阵，提取不含对角线的下三角，再拼接第一个投影向量。存在稠密分支时它是第一个向量；否则按 Embedding 列表顺序，再接序列分支。

行为序列编码配置：

| 字段 | 含义 |
| --- | --- |
| `vocab_size` | 独立行为序列 Embedding 表的词表规模 |
| `embedding_dim` | 序列 Embedding 维度 |
| `hidden_size` | 编码器隐藏维度 |
| `num_heads` | attention 头数，必须整除 `hidden_size` |
| `ffn_size` | FFN 中间维度 |
| `num_layers` | 编码层数 |

编码器采用完整双向 self-attention，显式物化 attention score 矩阵；每层为 Q/K/V 投影、attention、输出投影、残差与 LayerNorm、两层 FFN、残差与 LayerNorm，最后沿序列做 mean pooling。没有自回归生成阶段。序列表与其他 Embedding 表不共享权重；位置编码、mask、变长 packing、融合 attention、自定义算子和共享专家等需通过扩展 `graph.py` 适配。

## NPU 配置

| 字段 | 含义与默认值 |
| --- | --- |
| `name` | 硬件标识 |
| `compute_tops` | 按 dtype 提供 `matrix` 和 `vector` 两类算力，单位为每秒万亿次操作 |
| `memory_bandwidth_gbps` | 外存带宽，十进制 GB/s |
| `memory_capacity_bytes` | 设备外存容量，单位 bytes |
| `on_chip_memory_bytes` | 片上存储容量，默认 `0` |
| `on_chip_bandwidth_gbps` | 缓存命中 Embedding 行的片上读取带宽，默认 `0` |
| `matrix_alignment` | 每个矩阵乘法 M/K/N 的计算维度对齐粒度，默认 `1` |
| `compute_efficiency` | 所有执行单元吞吐乘数，范围 `(0,1]`，默认 `1` |
| `bandwidth_efficiency` | 外存带宽乘数，范围 `(0,1]`，默认 `1` |
| `random_access_efficiency` | Embedding lookup 外存带宽的额外乘数，默认 `1` |
| `launch_overhead_us` | 每个算子的额外启动开销，默认 `0` |
| `supported_ops` | 支持的算子列表，省略时假定支持所有当前建模的算子 |

算子类型为 `embedding`、`matmul`、`relu`、`concat`、`gather`、`softmax`、`add`、`layernorm`、`mean`、`sigmoid`。实际硬件不支持的类型应从 `supported_ops` 中删除，不会自动假定回退到 CPU。

统一按 1 MAC = 2 operations 计数。填写浮点算力时，TFLOP/s 与这里的数值口径相同；填写整数 TOPS 时应先确认厂商的 MAC 计数口径。`matrix_alignment` 仅估算执行的额外计算量，未模拟编译器的具体 tiling 或内部 padding 存储。

## 工作负载

```json
{
  "batch_size": 128,
  "lookups_per_sample": {"user_id": 1, "item_id": 1, "category": 4},
  "sequence_length": 0,
  "embedding_cache_hit_rate": 0
}
```

- `batch_size` 是一次模型执行中的评分样本数。多个请求或多个候选需要先映射为模型实际处理的样本维度；工具不模拟在线请求调度。
- `lookups_per_sample` 必须与模型中的 Embedding 特征名称完全一致，每项为正整数。多值特征采用 mean pooling。
- 存在序列编码器时，`sequence_length` 必须为正整数；否则必须为 `0`。
- `embedding_cache_hit_rate` 范围为 `[0,1]`，默认 `0`，统一用于普通与序列 Embedding lookup。大于零时必须提供片上容量和带宽。

缓存命中率是外部输入的假设，不从访问分布推导。当前只检查片上容量能否容纳单个 Embedding 行，不能证明给定容量能达到指定命中率。随机访问导致的带宽折损通过 `random_access_efficiency` 表达。未知命中率时应保留默认零值，并根据 profiling 校准带宽。

## Roofline 与内存模型

对每个算子，计算执行操作数 `F`、外存搬运字节数 `D_ext`、缓存读取字节数 `D_cache`：

```text
P_eff = 对应 dtype 和执行单元的吞吐 × compute_efficiency
BW_eff = 外存带宽 × bandwidth_efficiency
Embedding lookup 的 BW_eff 再乘 random_access_efficiency

t_compute = F / P_eff
t_external = D_ext / BW_eff
t_cache = D_cache / 片上带宽
t_op = max(t_compute, t_external, t_cache) + launch_overhead
t_batch = sum(t_op)
```

矩阵乘法使用矩阵单元，其余计算使用向量单元。MatMul 按对齐后的 M/K/N 估算执行操作数，另存未 padding 的有效操作数与利用率。只搬运数据的算子可以具有零操作数，但仍占用带宽。

每个未融合算子假设从外存读取输入和所需权重一次、写回输出一次，不假设跨算子缓存复用。Embedding 表不会被整个读取：其行数据流量按 `batch × lookup 数量 × dim × dtype_bytes` 计算，缓存命中的行数据转为片上读取；int64 索引读取和查表输出写回仍计入外存流量。

向量运算采用简化操作计数：ReLU/残差加法每元素 1 次，Sigmoid 5 次，含缩放的 Softmax 6 次，LayerNorm 7 次；mean pooling 按输入元素数计数。这些是解析近似，未逐项建模特殊函数吞吐、归约同步和精度转换。

峰值内存为全部常驻权重加上按算子顺序计算的峰值存活张量容量；含输入索引和任务输出，使用独立输出缓冲区，最后一次使用后释放。暂不包含编译器 workspace、内存碎片和运行时额外占用。累计搬运量不能作为内存容量。

片上路径目前只覆盖 Embedding 缓存命中，不是完整的分层 SRAM Roofline。模型假设算子内部计算与搬运理想重叠，算子之间串行，暂未建模有限 SRAM 引发的分块重复读取、DMA 竞争及启动同步。所有效率系数为 1 且启动开销为零时，输出标记为**给定假设下的理想耗时下界**；否则标记为经系数调整的解析估计，不等价于真实实测结果。

标准 Roofline 定义可参考 [NERSC 文档](https://docs.nersc.gov/tools/performance/roofline/)。

## 搜索配置

`parameters` 是“配置路径 → 候选值列表”的映射，索引从 `0` 开始。可变字段仅限：

- `embeddings.<index>.dim`
- `dense_mlp`，用完整宽度列表同时表达层数和各层宽度
- `interaction`，用完整对象切换 concat/dot 并保持依赖字段一致
- `towers.<index>.hidden_sizes`
- `sequence_encoder.embedding_dim`、`hidden_size`、`num_heads`、`ffn_size`、`num_layers`

仅已有序列编码器支持其结构搜索。输入特征、Embedding 词表规模、任务名称与输出维度等不能作为搜索字段。多个序列字段组合产生的不合法结构会记录为 `invalid`；工具不会静默修正用户的候选值。

| 字段 | 含义与默认值 |
| --- | --- |
| `method` | `adaptive`（默认）、`random` 或 `grid` |
| `seed` | 非负随机种子，默认 `0` |
| `max_evaluations` | 最大候选尝试数，含基线及拒绝候选，默认 `50` |
| `exploration_probability` | adaptive 策略随机探索概率，默认 `0.25` |
| `constraints.parameter_ratio` | 相对基线参数量区间，默认 `[0.9,1.1]` |
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
