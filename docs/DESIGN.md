# Kernel 调优第一版需求与设计

## 固定语义

输入 PyTorch `matmul.py` 或 Elementwise 脚本，提供固定 shape/dtype 的显式描述。第一版不搜索输入形状、模型架构或计算语义。MatMul 描述为二维 m/n/k；Elementwise 为 numel、input_count、flops_per_element。转置、布局、融合、batch 等需另外扩展模板与适配器。

Baseline：原始 PyTorch → ONNX → ATC → baseline OM。候选：版本匹配的官方开源/自定义算子源码 → Host tiling 与 Device 参数注入 → 编译注册 → 自定义算子图 → ATC → candidate OM。PyTorch 只做正确性参考。

性能比较仅使用两侧 OM 目标任务的 profiling 耗时中位数。`Task Duration(us)` 包含调度和完成响应，不是纯 Device 计算时间。转换、编译和加载不纳入适应度。

## 搜索分工

规则 Agent 分析 Roofline，决定优先参数；结合实测优胜者在原模板能力域内细化范围，停滞两代后扩展。遗传算法使用锦标赛选择、均匀交叉、偏置变异、精英保留和全域移民，避免精细化丢失探索。失败候选计入预算，重复配置不重测。预算仅计搜索评测，不含基线与最终复测。

MatMul 域：tile_m/tile_n/tile_k/buffers/block_dim。Elementwise 域：tile_elems/buffers/block_dim。所有参数必须来自实际模板能力。当前合法性检查包含对齐、buffer 数、block 上限与保守本地存储估计。实际 L0/L1/UB 各级容量、布局、尾块和参数依赖必须由真机适配器进一步校验。

Roofline 区分数学最低流量与 profiling 测得流量。只由 shape 得到的瓶颈是理论假设；不能将理想下界视为预测或实测耗时。配置中的 peak_flops 必须对应具体 dtype、执行单元和稠密/稀疏口径。

## 评测与验收

每个候选依次完成 build、export、atc、verify、profile，未通过者不得参与排名。ATC 响应须与源码/参数哈希一致，并提供实现 ID 和证据文件；verify 与 profile 响应须指向同一 OM 哈希。适配器必须证明实际选中候选，哈希和证据文件存在本身不能证明 ATC 的语义行为。

候选正确性以 PyTorch 输出为准，遵守配置 atol/rtol，覆盖实际输入和尾块。目标名称精确匹配，CSV 预先排除预热及非目标样本；不能按模糊名称混合多个节点、设备或 stream。

优胜者与 baseline 交错重新构建和复测。只有收益超过最小阈值和观测相对波动范围才采用候选；复测失败时保留 baseline。这是保守工程启发式，不是置信区间。

## 来源与复现

记录芯片、CANN 版本、源码标签、源码哈希、适配器文件哈希、所有阶段请求/响应和产物。候选安装到独立 OPP 路径，不覆盖 stock baseline 环境。

开源源码未必对应 stock OM 的实现；无法确认一致时，额外测量未修改源码实现，区分实现替换收益与调参收益。当前不自动完成该第三路对照。

当前在无真机环境下仅验证模拟流程，尚未提供特定芯片/CANN 的可部署 Ascend C 模板或真机采集实现。明确提供接入边界，不伪造上板性能。
