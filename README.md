# ParetoPilot

固定 shape/dtype 的昇腾算子调优工具，目标环境为 **Linux、Ascend 910B2、CANN 8.5.0**。
默认 MatMul 路线使用固定版本的 **BatchMatMulV3 完整源码**建立基线，再调优同一源码的 Host tiling。
内置 BatchMatMulV2 只作性能参照。保留 Roofline、Dummy Agent，不需要 Agent 服务或 API Key。

流程：PyTorch 正确性参考与 ONNX 导出 → stock 性能参照 → 默认 V3 源码基线 → 真实平台与 shape 生成参数域 → Roofline →
Dummy Agent 调整参数域 → 历史参数复测/遗传搜索 → 同源候选 → 三路交错复测 → 报告与结果缓存。
Add 新增 CANN 8.5.0 官方仓库的实验性 AddV2 同源调优路线，并提供 stock Add 安装源码定位入口；二者不混称同一实现。

## 支持范围

| 用例 | 输入 A | 输入 B | 输出 | 配置 |
| --- | --- | --- | --- | --- |
| 二维 MatMul | `[256,256]` | `[256,256]` | `[256,256]` | `examples/ascend910b_batchmatmul.json` |
| 共享权重投影 | `[32,100,128]` | `[128,256]` | `[32,100,256]` | `examples/ascend910b_batchmatmul_shared_rhs.json` |
| 独立候选打分 | `[32,1,128]` | `[32,128,100]` | `[32,1,100]` | `examples/ascend910b_batchmatmul_batched_rhs.json` |

以上用例均为 FP16、外部 ND、无转置和 bias，均接入 V3 源码调优；支持示例中的非对齐维度与
批次广播。Host 参数包含 `core_limit`、`tiling_policy`、`batch_strategy` 和 `base_m/base_n`。
显式分块使用官方 `SetFixSplit(M,N,-1)`，baseK 自动计算；`base_m=base_n=0` 保留官方自动策略。
三个用例的 kernel 直调烟测均检查默认参数和一个显式分块候选，详见 [直调说明](docs/KERNEL_DIRECT.md)。
示例默认启用 `search.domain_mode="shape"`：修改 operator 中的 shape 后自动重建参数域，无需手动修改 space。
相同 shape 的历史最优参数先复测，相近 shape 只作预热候选；最终仍执行新的 OM 确认。
用 `--fresh-search` 忽略历史参数并保留编译缓存；用 `search.domain_mode="manual"` 恢复静态参数域。
AddV2 示例为 FP16、相同输入 shape、`numel=1048576`，要求元素数为 16 的倍数，以保留原始 Device 源码。
参数是 `core_limit`、`tile_elems`，默认均为 0（官方默认推导）；直调按 profiler 目标任务 P50 排名，最终走 OM 确认。
旧 PpAdd 模板及非对齐尾块路径仍可显式选 `--add-implementation bundled`。
stock Add 路线先测量并收集安装包实际编译分支，尚未宣称已支持其源码 tiling 寻优。命令与边界见 [Add 两条路线](docs/OFFICIAL_ADD.md)。

官方源码与离线构建依赖已随 GitHub 仓库打包。**本地没有 CANN/NPU；源码编译、精度和性能仍需在
910B2 上验收。** 构建依赖、源码适配范围和控制点见 [BatchMatMulV3 源码调优](docs/OFFICIAL_BATCHMATMUL.md)。

## 在 910B2 上测试

首次确认三个用例的源码均能编译运行：

```bash
cd /home/ma-user/work/ParetoPilot
git pull --ff-only origin main
/home/ma-user/anaconda3/envs/PyTorch-2.6.0/bin/python3 scripts/test_910b.py \
  --cann-path /usr/local/Ascend/cann-8.5.0 \
  --soc Ascend910B2 --device 0 --deps check \
  --operator matmul --matmul-implementation batch-v3 --matmul-case all \
  --scope smoke --timeout 3600
```

把 `--scope smoke` 改成 `--scope full`，执行三个用例的完整 Host tiling 搜索。
仅跑一个用例可用 `--matmul-case 2d`、`shared_rhs` 或 `batched_rhs`；`3d` 选择两个三维用例。
`--matmul-implementation` 默认 `batch-v3`，`--matmul-case` 默认 `2d`。
脚本自动加载 CANN 环境，`--deps check` 不安装依赖。无需 sudo。
完整安装/离线安装选项见 [部署](docs/DEPLOYMENT.md)。

smoke 每个 shape 只运行一次默认 V3 的正确性和 profiling 检查，保留 Roofline；不运行 V2、搜索或确认轮次，不声明调优收益。仅选 MatMul 的 smoke 也跳过单元测试和模拟闭环；full 保留完整验证与 Dummy Agent 搜索。
full 的候选必须相对默认 V3 的收益超过最小阈值及观测噪声，并有稳定、不同的实际 tiling，才接受收益。
即使候选仍慢于 V2，也会如实报告 V3 相对自身源码基线的收益；V2 不参与源码实现选择。
没有收益时选择 `source_baseline`。

## 结果

每次运行创建 `runs/910b-*`，不覆盖已有结果。`test-summary.json` 汇总所选用例，阶段结果为：

- `matmul-smoke` / `matmul-full`
- `matmul-shared-rhs-smoke` / `matmul-shared-rhs-full`
- `matmul-batched-rhs-smoke` / `matmul-batched-rhs-full`

每个阶段有 `summary.json`、`report.html`、`report.md` 和 `roofline.svg`；搜索阶段另有 `trials.csv`。
`baseline/` 是 V2 性能参照；`source_baseline/` 是默认 V3 源码构建；`trials/` 是候选；
`confirmation/` 是交错复测。各源构建保存源码/适配/安装哈希和实际 tiling。
BatchMatMulV3 默认跨参数、shape 和运行复用经哈希校验的源码 OPP（项目 `.cache/official-builds`）。Host 在每次 ATC 进程读取参数，因此调整现有参数无需重新构建 OPP。源码、适配器、CANN 记录/依赖或编译工具变化会触发重建；修改 C++ 策略逻辑仍需构建一次。每次仍重新 ATC 编译 OM、校验和采样，使用独立编译缓存。

可用 `--build-cache /home/ma-user/work/paretopilot-cache` 指定缓存，或 `--no-build-cache` 关闭缓存。首次构建仍可能较慢；复用只消除 configure/build/package/install，不消除 ATC 和测量耗时。

关键复测字段：`confirmation.primary_baseline_us` 是默认 V3 耗时，`candidate_us` 是候选耗时，
`primary_speedup` / `source_speedup` 是相对 V3 的加速比，`stock_reference_us` / `baseline_us` 是 V2 耗时，
`stock_speedup` / `speedup` 是相对 V2 的加速比。单位均为目标任务微秒，阶段 `seconds` 包含构建等开销。
`status=passed` 表示验收通过，不等于 `tuning_accepted=true`。

Roofline 对共享权重只计一份权重流量，独立 batch 权重按 batch 计数；这是形状流量下界。
算力/带宽示例值是配置假设，需按实际硬件和 dtype 替换，不代表实测规格。
Dummy Agent 原样返回参数域，每代保留 Roofline/历史输入和建议输出。

## 本机模拟与验证

```bash
python -m paretopilot run --config examples/simulation_matmul.json --output runs/sim-matmul
python -m paretopilot run --config examples/simulation_add.json --output runs/sim-add
python -m unittest discover -s tests -v
```

模拟数据不能证明硬件收益。测试覆盖控制器、源码适配、实际 PyTorch→ONNX 导出和 PyACL mock；
依赖缺失时相关测试明确 skip。mock 不证明 Ascend kernel 可以编译或运行。

## 扩展与旧流程

- [Agent 接口](docs/AGENT.md)：可替换本地 `propose(context)`，默认 Dummy Agent 无服务依赖。
- [后端](docs/ADAPTER.md)：构建、正确性与 profiling。
- [旧二维 MatMulV3](docs/OFFICIAL_MATMUL.md)：使用原 `examples/ascend910b_matmul.json`。
- `--matmul-implementation legacy` 保留旧二维 MatMulV3 和三维 baseline-only 行为，以及同算子发现门禁。
- `scripts/probe_bmmv2_strategy.py` 保留为独立 V2 编译策略实验，不属于当前 V3 Host tiling 流程。
- `python -m paretopilot package --output dist/ParetoPilot.zip` 可生成源码包；测试机优先通过 GitHub 更新。

## Kernel 直调寻优（可选）

可使用 `python3 -m paretopilot run --execution kernel --config examples/ascend910b_batchmatmul.json --output runs/kernel-tune`。
首次编译官方 FP16/ND kernel 变体；搜索阶段由常驻进程调用官方 Host tiling 并直接执行 kernel，无需逐候选 ATC。
直调使用设备 Event 区间 P50，最终收益仍由 ATC/OM profiling 复测确认，两种计时分别标记。
当前完成 CPU/ABI mock 验证，910B2 实际编译与运行需要上板 smoke 确认。三个 shape 的命令、编译边界和日志见 [Kernel 直调](docs/KERNEL_DIRECT.md)。
