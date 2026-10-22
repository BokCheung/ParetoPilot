# 本地验证记录

日期：2026-10-22。环境：Windows、Python 3.12.14。

`python -m unittest discover -s tests -v`：24 项全部通过，无跳过。

- 核心：配置约束、Roofline 公式和流量口径、Dummy 不调整参数域、插件域验证。
- 搜索：固定 seed 可复现、参数去重和预算、失败候选、噪声/小收益回退、已有输出保护。
- 采样：精确筛选、中位数、重复导出、混合节点、NaN 拒绝。
- 原生链路合同：构建参数注入、独立 OPP、参数签名缺失、OM 被替换时拦截。
- PyACL mock：预热排除、ND padding、错误输出停止、设备调用失败清理。
- 实际 CPU 导出：内置 MatMul、Add 和外部 Python 基线的 PyTorch→ONNX 导出及 ONNX schema 检查。
- 子进程：超时终止及命令日志；修复了 Windows venv 启动器的子进程残留问题。

本地导出测试使用 torch 2.14.1、onnx 1.23.2、numpy 2.5.4；这些是本机验证版本，不是目标 CANN 环境的强制依赖版本。目标机器优先使用兼容其 Python/PyACL/ATC 的组件组合。导出固定使用 ONNX opset 13 和 TorchScript 导出路径。

另外已跑通 MatMul、Add 两套模拟搜索，每套 12 个候选，生成完整报告。模拟数据与 PyACL mock 不代表 NPU 计算结果。

**尚未验证**：910B 上的 Ascend C 编译、OPP 注册、ATC 自定义节点转换、真实 PyACL 推理/精度和 msprof CSV，以及硬件性能收益。交付中含完成这些阶段所需的源代码和启动命令，按 DEPLOYMENT.md 先运行 smoke 验收。
