# Kernel 调优设计（离线 Dummy Agent 版本）

## 固定语义

本版本针对 Ascend 910B / CANN 8.5.0：二维 FP16 MatMul，或同 shape 的两输入 FP16 Add。固定输入 shape/dtype，搜索不改变计算语义或模型架构。MatMul 的 M/N/K 要求 16 对齐；Add 支持尾块。其他 Elementwise 操作、转置、batch、融合和不同 dtype 需增加模板。

Baseline：内置或用户 PyTorch build_case → ONNX → ATC → baseline OM。候选：官方 MatMulV3 Host/Device 源码 → 参数入口 → 官方 CMake 编译 → 独立 OPP → 自定义 ONNX → ATC → candidate OM。Add 仍使用内置模板与 msopgen。无需用户已有工程或 Agent 服务。

性能比较仅使用 OM 目标任务 Task Duration(us) 中位数，不包含导出、编译、加载或 CPU PyTorch 耗时。PyTorch 只生成固定输入的正确性参考。

## Roofline 和 Dummy Agent

Roofline 保留 FLOPs、数学最低流量、可选实测流量、算术强度、理想时间下界、瓶颈假设及建议参数优先级。shape 最低流量不冒充实测 HBM 流量，理想下界不冒充耗时预测。硬件峰值需按具体型号、dtype、执行单元和稠密/稀疏口径填写，示例值仅作假设。

每代 Roofline 与完整历史传给 Agent.propose(context)。默认 DummyAgent 返回原始参数域及空优先级，不作域调整，不调用模型或网络。真实策略留给后续插件实现，见 AGENT.md。

## 搜索

锦标赛选择、均匀交叉、逐参数变异、精英保留和随机未测配置注入。配置 seed 固定随机序列。测过的参数不重复评测，失败候选计入预算；预算不包含首次基线与最终复测。精英参与繁殖而不重复计费。

合法性预筛包括对齐、buffer 数、block 上限和保守存储估计；实际资源依赖由 Host 和 CANN Tiling API 进一步验证。域受限时可在未耗尽预算前结束。搜索停在 generations、budget 或无未测合法配置任一条件。

官方 MatMul 搜索 core_limit、tiling_policy，基于官方 tiler 联合计算分块和缓冲，详见 OFFICIAL_MATMUL.md。Add 搜索 tile_elems、buffers、block_dim。

## 测量与选择

PyACL 进程加载并执行同一 OM：先正确性检查，再预热，随后开启 profiling、执行 repeats 次、停止采样，再复核输出。资源通过 ExitStack 清理。当前使用一组固定 seed 的输入；这不是对所有输入的形式化正确性证明。

CSV 按精确字段匹配，要求恰好 repeats 条目标任务，并拒绝跨名称、设备、stream/model 混样；融合、多任务拆分和重复导出均不能悄悄混入排名。NaN、非正耗时、错误精度、编译或运行失败均记为失败。

官方 MatMul 使用独立 ONNX 源名称 ParetoPilotOfficialMatMul，经 parser 映射到隔离 OPP 中的 MatMulV3，并要求官方源码提交签名。Add 使用 PpAdd；其 Host tiling 输出参数哈希签名，控制器要求在 ATC 日志中找到该签名，再记录源码、安装产物、OM 和日志哈希。它证明所编译 Host 被调用，最终还需实际正确性和目标任务记录共同验收，不单靠“证据文件存在”。

优胜者与 baseline 每轮重新构建/转换，执行顺序轮流交换。只有中位数相对收益大于 min_gain 和两组轮次中位数的最大相对极差才采用候选。该规则是工程启发式，不是统计置信区间。复测失败或收益不足保留基线，selected_params=null。

## 可复现性与边界

保存配置、环境、模板哈希、生成源码哈希、OPP 哈希、OM 哈希、命令、日志、输入输出和 Agent 决策。不会覆盖已有试验目录，不支持自动续跑。

官方 MatMul 增加 source_baseline 第三路对照，交错顺序为 baseline/source_baseline/candidate 及逆序；分别报告相对内置与官方默认源码的收益。candidate 必须同时超过两个对照的收益和噪声阈值；仅默认源码优于内置时可选择 source_baseline。相同默认参数的重复构建不会计为调优收益。Add 保留原先两路对照。

已提供原生编译与运行链路；本地没有昇腾硬件，真机编译、精度和性能仍须按 DEPLOYMENT.md 的 smoke 步骤验收。
