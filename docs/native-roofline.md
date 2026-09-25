# 原生搜推广模型的逐模块 Roofline

`examples/ads_model_config.jsonnet` 中每个节点都有计算记录，包含公式、输出 shape、内部参数/状态量、算子明细、计算量、访存量、算术强度及预测耗时。`sfps_feature_embedding` 为排除节点；split、squeeze、flatten_list 的设备计算与流量为零，但它们影响依赖与张量生命周期。

## 运行与输出

```bash
python -m paretopilot validate --model examples/ads_model_config.jsonnet --npu examples/npu.json --workload examples/ads_workload.json
python -m paretopilot analyze --model examples/ads_model_config.jsonnet --npu examples/npu.json --workload examples/ads_workload.json --output runs/ads-roofline
```

输出 `report.md`、`analysis.json`、`modules.json`、`module_types.json` 和 `modules/1.json` 至 `modules/83.json`。序号对应配置节点顺序，JSON 内保留节点名称。逐节点 JSON 自带底层算子的 shape、计算/访存时间，不需要另外定位总报告。重复同一实验可加 `--resume`；修改任意输入后使用新目录。

示例硬件和 workload 都是演示数据。示例采用 batch=128、49 个就绪特征、10 个连续特征，序列统一长 20，特征所属 region 是演示分组。这些数量、长度和分组不是从业务配置完整恢复出来的；业务配置本身没有完整的特征 schema。

## 输入张量与引用

原生 workload 独立于组合式 workload，主要字段如下：

| 字段 | 含义 |
| --- | --- |
| `batch_size`, `dtype` | 一批评分样本数与统一数据精度 |
| `ready_features` | SFPS 之后已在设备内存中的张量，每项包含 `name`、`shape`、`region` |
| `dense_features` | 模型内 log_dense 的连续输入张量，字段同上 |
| `shape` | **不含 batch 维**。例如 `[20,40]` 表示 `[B,20,40]`，`[1,24]` 表示 `[B,1,24]` |
| `region` | 字符串列表，用于配置中的 `"sence_slot" in region` 等选择器 |
| `kind` | 就绪特征可选 `single_discrete` / `multiple_discrete`；默认按首维是否为 1 推断 |
| `module_options` | 按模块 type 覆盖近似模型选项 |
| `node_options` | 按节点 name 覆盖选项，优先级高于 `module_options` |
| `outputs` | 可选的最终输出引用列表；默认最后一个节点的输出 |
| `synthetic` | 示例设为 true，报告明确提示数据仅用于演示 |
| `description` | 数据来源或适用范围说明 |

`ready_features` 直接决定外部输入 shape，SFPS methods 不用于推断或估计查表行为。若实际返回的是已聚合向量，需让 shape 与模型内的 postprocess 操作一致，避免重复 pooling。原始配置要求的 split、DIN 和下游宽度会执行一致性检查。

支持静态对象/数组、注释、尾逗号，以及 `local model_structure = [...]; model_structure`。也可输入 JSON 数组或含 `model_structure` 的 JSON 对象。动态 Jsonnet 需先自行导出 JSON，不加载 import，也不执行函数。

引用支持节点输出、列表索引、特征名、`values(...)`、`name/type/region` 的 `==` / `!=` / `in` / `not in` 条件及 `and/or/not`。只解析允许的 AST 节点，不使用 `eval`。空选择、未定义/前向引用、未知模块和不兼容 shape 会报错，不跳过或伪造缺失输入。节点需按依赖顺序排列；当前按全部声明节点执行一次，不做编译器的死分支消除。

## 统一计算口径

设 `B` 为 batch，`L` 为序列长度，`D` 为通道维，`T` 为 token 数，`s` 为每元素字节数。MAC 按两次操作计数。每个底层算子独立计算：

```text
F_useful   = 按逻辑 shape 计算的操作数
F_executed = matrix: 2 * groups * align(M) * align(K) * align(N)
             vector: F_useful
bytes      = sum(input_tensor_bytes) + output_tensor_bytes + weight_bytes_read
AI         = F_executed / bytes
t_compute  = F_executed / effective_engine_ops_per_second
t_memory   = bytes / effective_bandwidth_bytes_per_second
t_operator = max(t_compute, t_memory) + launch_overhead
t_module   = sum(t_operator)
t_e2e      = sum(t_module)
```

视图的 bytes、计算量和启动开销均为 0；其下游读取按实际 slice shape 计流量，容量按完整源张量的存活期计。concat/stack/transpose 显式物化。混合矩阵、向量及访存算子的模块不能将 F/bytes 聚合后再取一次 max；报告中的模块 AI 仅作汇总指标。

## 各模块的公式与假设

以下是计算模型定义，不是从业务实现中验证出来的内核。特别是 CAN、Token Mixer、HR Tower，仅凭类型名和维度无法确定实际计算路径。

| 模块 | 拆解与 useful ops |
| --- | --- |
| `sfps_feature_embedding` | 输入绑定，F=0、bytes=0、参数=0；排除 SFPS 表、通信和缓存 |
| `feature_sequential` / `sum` / `mean` / `pooling` | 按声明轴归约；sum 为 `output_elements*(L-1)`，mean 为 `output_elements*L`；3D pooling 流量为 `s*B*D*(L+1)` |
| `split` / `squeeze` | 零成本视图，保留源存储；不生成拷贝或启动开销 |
| `flatten_list` | 只展开列表容器；不 flatten 张量，不生成设备算子 |
| `concatenate` / `stack` | F=0，流量为所有输入加输出；单输入 concatenate 直通 |
| `dense` | `2*M*K*N` 矩阵运算，加 bias 每输出 1 次操作，再做配置激活；`M=prod(input_shape[:-1])` |
| `dnn` | 各 Dense 相加；默认所有层 ReLU，bias 开启 |
| `loop` | 对每个输入分支依次执行 layer_seq 内的 concatenate/dense；每支权重独立 |
| `din` / `din_v2` | 广播 target，构造 `[q,k,q-k,q*k]`，执行 `4D -> hidden_dims` 打分 MLP，分数乘 sequence，再 sum；矩阵 F 为各层 `2*B*L*d_i*d_(i+1)` 之和 |
| `ads_can` | 假设 sequence embedding 编码每样本动态 MLP 的矩阵，target 为输入。矩阵 F 为每对特征 `2*B*L*sum(d_i*d_(i+1))`；逐层 tanh，序列 sum，最后拼接 |
| `epnet` | 拼接 domain 和 feature，执行 gate MLP，`2*sigmoid(gate)` 乘 feature；默认隐藏宽度 `[64]` |
| `gatenu` | conditioning 输入执行 gate MLP，`2*sigmoid(gate)` 乘第二个 feature 输入；默认隐藏宽度 `[64]` |
| `token_mixer` | 假设沿 token 维做所有通道共享的 `T -> T` 线性变换；F=`2*B*D*T*T`，矩阵参数 `T*T`；前后 transpose 各计一次搬运 |
| `per_token_ffn` | 假设每 token 独立 `D -> H -> D`、中间 ReLU、可选残差及末尾 LayerNorm；矩阵 F=`4*B*T*D*H`，默认参数 `2*T*D*H`，默认 H=4D |
| `hr_tower` | **代理结构**：拼接 feature 和 domain，执行 `(D+C) -> hr_layer_dim -> output_dim` MLP，默认 output_dim=1；没有推断真实 HR 专家、路由或超网络 |
| `log_dense` | **近似结构**：每连续特征独立做对数分桶及本地表 lookup；默认每标量 5 次向量操作，表容量为 `hash_num*embedding_size*s`，访存只计所选条目 |
| `string_expression` | 仅支持 `x/(x+(1-x)/y)`：减、除、加、除，每输出共 4 次操作；y 为正的常量 |

DIN 的目标相关局部激活思路参考 [DIN 原论文](https://arxiv.org/abs/1706.06978)，此处的 `[q,k,q-k,q*k]` 拆解是所选近似配置。默认不做 softmax；可显式打开。一个 target 当前对应一个 sequence feature，不将任意多字段序列静默合并。`use_same_dnn` 控制同节点内打分网络权重和 DICE 状态是否共享，读取流量仍按每次调用计。

CAN 示例选择 bias-free `8 -> 3` 动态层，以匹配 sequence embedding 宽度 24。**24=8×3 仅是可配置假设，不能证明业务 CAN 的真实结构**。允许多层 `hidden_sizes`，但编码宽度必须等于各矩阵元素数之和。当前单阶、只取末层输出，不含高阶交互、动态 bias 或隐含额外投影；动态矩阵来自就绪输入，因此不重复计为模型内部静态参数。

GateNU/EPNet 的 gate 结构、Token Mixer 的可学习 token 变换、PFFN 的扩展比与参数共享都需要在获得实际定义后校准。`log_dense` 不模拟 log_base 或 rate_feature_list 的数值效果，分桶、截断及 hash 的性能合并为 `bucket_ops`；该局部表是模型内部状态，属于当前边界。

推理阶段 dropout 和 stop_gradient 不生成运算。DICE 采用固定统计量，计每元素 11 次近似操作及每通道 3 份状态；LayerNorm 每元素 7 次，sigmoid 5 次，tanh 6 次。特殊函数吞吐、推理框架的融合与布局拷贝可能显著改变真实耗时，当前未执行框架或 NPU 内核。

## 调整近似结构

以 workload 为入口，不改业务 Jsonnet 中的训练与服务参数：

```json
{
  "module_options": {
    "din": {"normalize_weights": false, "use_bias": true},
    "din_v2": {"normalize_weights": false, "use_bias": true},
    "ads_can": {"hidden_sizes": [3], "activation": "tanh"},
    "epnet": {"hidden_sizes": [64]},
    "gatenu": {"hidden_sizes": [64]},
    "token_mixer": {"hidden_sizes": [], "use_bias": true},
    "per_token_ffn": {"hidden_size": 1664, "shared_weights": false},
    "hr_tower": {"output_dim": 1},
    "log_dense": {"bucket_ops": 5}
  },
  "node_options": {
    "pffn2": {"hidden_size": 832},
    "gate1": {"hidden_sizes": [32]}
  }
}
```

Dense/DNN、DIN、门控、Mixer、PFFN、HR Tower、loop 支持 `use_bias`。PFFN `shared_weights=true` 时矩阵参数降为 `2*D*H`，矩阵分组从独立 token 变为共享权重，padding 也随之重算。Token Mixer 的 hidden_sizes 位于最终 T 宽度层之前。未支持的选项会报错。

Python API：

```python
from paretopilot.config import load_json
from paretopilot.native_config import load_model
from paretopilot.native import evaluate_native

result = evaluate_native(
    load_model("examples/ads_model_config.jsonnet"),
    load_json("examples/npu.json"),
    load_json("examples/ads_workload.json"),
)
for module in result["modules"]:
    print(module["name"], module["metrics"]["latency_ms"])
```

原生图目前支持校验与单点评估，不接入自动结构搜索。代码入口为 `native_config.py` 的配置/引用解析、`native.py` 的 `PROFILES` 和 `NativeBuilder.lower`，所有模块共用 `roofline.py` 的算子计算内核。
