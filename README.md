# ParetoPilot

固定 shape/dtype 的昇腾算子 Kernel 调优 Agent。输入 PyTorch baseline 与可调算子源码，基于 Roofline 分析搜索方向，通过遗传算法优化参数，统一比较 **baseline OM 与 candidate OM 的目标算子 profiling 任务耗时**。

本项目已按 Kernel 调优需求重建，不再包含模型结构搜索、推荐模型配置、硬件对比或参数扫描功能。

## 评测流程

```text
PyTorch baseline → ONNX → ATC → baseline OM → 校验与 profiling
                                          ↓
                       Roofline → Agent 制定/调整参数域
                                          ↓
                       遗传算法 → 合法候选 → 编译自定义算子
                                          ↓
                       自定义算子图 → ATC → candidate OM
                                          ↓
                       校验、生效证据、profiling → 适应度
                                          ↓
                       基线/优胜者交错复测 → 参数与报告
```

PyTorch 输出只作正确性参照；直接执行 PyTorch 的时间不参与性能排名。适应度为目标算子 `Task Duration(us)` 的中位数。Baseline 与候选保持输入、硬件、精度、预热和采集条件一致。

## 快速运行

Python 3.10+，控制器只使用标准库。无需安装 PyTorch/CANN 即可验证模拟流程：

```bash
git clone https://github.com/BokCheung/ParetoPilot.git
cd ParetoPilot
python -m pip install -e .
python -m paretopilot validate --config examples/matmul.demo.json
python -m paretopilot analyze --config examples/matmul.demo.json
python -m paretopilot space --config examples/matmul.demo.json
python -m paretopilot tune-kernel --config examples/matmul.demo.json --output runs/matmul-demo
python -m paretopilot tune-kernel --config examples/elementwise.demo.json --output runs/elementwise-demo
python -m unittest discover -s tests
```

输出目录必须为空。示例的 OM、耗时、正确性及 ATC 记录全部为 **模拟数据**，不能作为实际性能结论。真实链路需要 Linux 昇腾环境、匹配的 CANN 源码和工程适配器，详见 [需求与设计](docs/DESIGN.md)、[适配器协议](docs/ADAPTER.md)。

## 代码结构

| 模块 | 职责 |
| --- | --- |
| `config.py` | 固定算子、硬件、离散参数域和运行配置校验 |
| `space.py` | 模板合法性、Roofline 参数优先级引导的域调整 |
| `roofline.py` | FLOPs、最低/实测流量、算术强度及瓶颈假设 |
| `search.py` | 锦标赛选择、交叉、变异、精英保留、移民、交错复测 |
| `backend.py` | build/export/atc/verify/profile 阶段执行与证据检查 |
| `profiling.py` | 精确匹配目标算子，导入 OM profiling 微秒样本 |
| `artifacts.py` / `report.py` | 原子记录、源码哈希、实验报告 |
| `adapters/ascend_driver.py` | 真机工程接入模板，未配置时明确失败 |
| `scripts/demo_adapter.py` | 仅模拟协议演示，拒绝 real 模式 |

## 输出

`manifest.json` 保存配置和来源哈希；`history.json` 保存去重试验；`baseline/`、`trials/`、`confirmation/` 保存各阶段请求、响应、日志与产物；`summary.json` 保存搜索域变化、Roofline、复测和决策；`best_params.json` 保存最终选择；源码快照用于复现；`report.md` 是人读报告。

## 当前边界

支持二维 MatMul 和单一 Elementwise 模板契约，不执行任意 Python 源码做自动语义推断。规则 Agent 已实现；尚未绑定 LLM。真实 Ascend C 源码、ONNX 导出、注册、ATC、OM 执行与 profiling 由匹配具体芯片/CANN 的适配器实现。契约 JSON 不是可部署 Kernel。没有断点续跑；失败日志保留在原输出目录。
