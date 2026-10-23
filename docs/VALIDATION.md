# 本地验证记录

日期：2026-10-22。环境：Windows、Python 3.12.14。

`python -m unittest discover -s tests -v`：48 项全部通过，无跳过。

2026-10-23 增量验证：MatMul 实机 CSV 显示 `/MatMul` 的类型为 `BatchMatMulV2`，已修正真实基线示例筛选条件。新增回归测试后，`python -m unittest discover -s tests -p test_profiling.py -v` 的 8 项全部通过；Ascend MatMul 配置 validate 通过。回放所提供的两条记录为 7.520/7.620 μs，同时确认旧类型得到 0 条、两条记录不能满足正式测试的 20 条要求。本次未在本机重跑完整硬件流程。

- 核心：配置约束、Roofline 公式和流量口径、Dummy 不调整参数域、插件域验证。
- 搜索：固定 seed 可复现、参数去重和预算、失败候选、噪声/小收益回退、已有输出保护。
- 采样：精确筛选、中位数、重复导出、混合节点、NaN 拒绝。
- CANN CSV 兼容：新增 7 项测试，使用测试机提供的 Add 表头和两条记录验证 `OP Type`、`Stream ID`、`Model ID` 等别名；Task Duration 中位数为 8.49 μs。同时验证新旧配置互通、混合标识拒绝、重复/缺失列诊断和 inspect-profile 输出。该 CSV 回放不代表完整上板流程已通过。
- 原生链路合同：构建参数注入、独立 OPP、参数签名缺失、OM 被替换时拦截。
- PyACL mock：预热排除、ND padding、错误输出停止、设备调用失败清理。
- 实际 CPU 导出：内置 MatMul、Add 和外部 Python 基线的 PyTorch→ONNX 导出及 ONNX schema 检查。
- 子进程：超时终止及命令日志；修复了 Windows venv 启动器的子进程残留问题。
- 一键测试脚本：新增 17 项测试，覆盖 CANN 新旧目录与精确版本检查、环境脚本路径传参、配置复制、在线/离线命令生成、阶段顺序、真实结果验收、失败即停、退出码、已有输出保护及诊断包筛选。CANN 环境加载和设备阶段使用 mock 验证控制逻辑；不是 Linux/CANN 实测。

本地导出测试使用 torch 2.14.1、onnx 1.23.2、numpy 2.5.4；这些是本机验证版本，不是目标 CANN 环境的强制依赖版本。目标机器优先使用兼容其 Python/PyACL/ATC 的组件组合。导出固定使用 ONNX opset 13 和 TorchScript 导出路径。

`scripts/test_910b.py --help` 和 Python 编译检查通过。源码包包含一键脚本及 requirements/910b.txt。脚本默认安装的 torch 2.7.1、onnx 1.16.1、numpy 1.26.4 组合尚未在本机安装验证；Linux 的 set_env.sh 加载、该依赖组合和硬件阶段仍须在目标测试机验收。

另外已跑通 MatMul、Add 两套模拟搜索，每套 12 个候选，生成完整报告。模拟数据与 PyACL mock 不代表 NPU 计算结果。

**本机尚未验证**：910B 上的 Ascend C 编译、OPP 注册、ATC 自定义节点转换、真实 PyACL 推理/精度、完整 msprof 采集导出链路，以及硬件性能收益。交付中含完成这些阶段所需的源代码和启动命令，按 DEPLOYMENT.md 先运行 smoke 验收。
