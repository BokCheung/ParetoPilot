# 910B2 BatchMatMulV3 完整源码调优

## 源码与架构

源码来自 CANN ops-nn 8.5.0 固定提交 `9f8a66e795ef1842c2118cf5bbadfe3624bdd1ef`，
位于仓库的 `paretopilot/vendor/cann_8_5_0/ops-nn-matmul.zip`，附上游许可证和逐文件清单。
构建依赖随 `build-deps.zip` 提供，不在测试机拉取开发依赖源码。

`implementation.kind=cann_batch_matmul_v3` 选择完整 BatchMatMulV3 模块，包含其 Host/图推导/Device
及依赖的 MatMulV3 和 matmul.common 模块。CMake 固定 `ASCEND_COMPUTE_UNIT=ascend910b`、
`ASCEND_OP_NAME=batch_mat_mul_v3`。生成的动态 Python adapter 必须包含普通 BatchMatMulV3 入口，
不能选择 `batch_mat_mul_v3_apt.cpp`。本项目不依据安装目录中某个同名文件来选择内核。

构建在运行目录生成私有 OPP，沿用上游 Device 源码，既不修改系统 CANN 源码，也不安装进系统 OPP。
保留安装版 `libophost_comm_legacy.so` 运行依赖的隔离副本。生成与安装的两个动态 Python adapter
`mat_mul_v3.py`、`batch_mat_mul_v3.py` 必须存在且哈希相同。

## 可审核的适配

- 沿用已有离线构建适配，移除当前路径不用的 protobuf/Eigen/abseil 下载依赖。
- 安装打包阶段枚举生成的动态 Python 文件，避免配置阶段空 GLOB 导致缺文件。
- 扩展上游的 dtype inference 注册，保留原 shape/range inference；当前只开放 FP16 输出。
- 自定义 ONNX 图节点 `ParetoPilotOfficialBatchMatMul` 映射到 GE `BatchMatMulV3`，固定无转置、无 bias。
- 修改 Host 参数入口及记录有效 tiling；Device kernel 与头文件字节保持上游原样。

每次构建在 `source_manifest.json` 保存上游提交、原档案哈希、每项补丁前后哈希、参数哈希、
Device/依赖源码哈希。`build_manifest.json` 还记录安装 OPP 和生成 adapter 的哈希。

## Host tiling 参数

| 参数 | 默认值 | 域与语义 |
| --- | --- | --- |
| `core_limit` | `0` | 示例 `[0,12,24]`；0 保留物理核数，非零在推导前限制核心预算，超过物理核心数拒绝 |
| `tiling_policy` | `0` | 0 保留上游知识库查询及启发式回退；1 跳过知识库查询，采用原启发式 |
| `batch_strategy` | `0` | 0 保留上游 L1/full-load/multibatch 自动选择；1 保留 common tiling，跳过 L1-full-load 与 multibatch 专门选择 |

默认参数只增加观测记录，不改变上游默认 tiling 算法。核心预算在父类 GetPlatformInfo 中注入，
strategy 在 BatchMatmulV3BaseTiling::DoLibApiTiling 的依赖计算前选择路径。不会在计算完成后
强改 baseM/N/K 或 tiling key；缓冲、同步、batch 广播和尾块处理继续由相应上游路径完成。

ATC 日志必须出现参数签名、源码提交和 `PARETOPILOT_BATCH_TILING`。该记录来自 BatchMatMulV3 自身
PostTiling，包含请求参数、最终 baseM/N/K、used_core_num、block_dim、batch_count、tiling_key、
multi_batch、load_mode。common 策略要求 multi_batch=0、load_mode=0，否则拒绝试验。

## 基线、参照与收益

`baseline/` 运行内置 ONNX MatMul，严格测量 `BatchMatMulV2`，只作性能参照。
`source_baseline/` 构建上述同一套源码的默认参数，实际执行 `BatchMatMulV3`，是调优基线。
`trials/` 只改变 Host 参数，其实际算子同样必须为 `BatchMatMulV3`。
三路都使用相同输入、PyTorch 参考输出、精度标准、预热和固定目标任务样本数。

Roofline → Dummy Agent 使用 V3 源码基线作为 context.baseline，另提供 stock_baseline。
确认阶段交错测量 V2、默认 V3、候选 V3，奇偶轮反转顺序。候选满足：

1. 参数不同于默认配置；实际 tiling 在同配置复测间稳定，并与默认 V3 不同；
2. 相对默认 V3 的收益严格大于 min_gain 与默认/候选观测相对极差；
3. 全部正确性、样本数、算子类型和源码签名检查通过。

满足则选择 candidate，否则保留 source_baseline。V2 即使更快，也不阻止报告相对 V3 的调优收益；
它同时作为性能参照展示，不能把 V3 相对自身的收益冒充超过 V2。

`confirmation.primary_baseline_us`、`primary_speedup` 表示 V3 默认基线与 V3 加速比；
`stock_reference_us`、`stock_speedup` 表示 V2 参照与 V2 加速比。旧兼容字段 baseline_us/speedup
仍表示 V2，source_baseline_us/source_speedup 表示 V3。

默认使用跨运行不可变源码包缓存，位于项目 `.cache/official-builds`。所有参数和三个 shape 共用同一源码包，
Host 从每次 ATC 进程的环境读取 core_limit、tiling_policy、batch_strategy 和请求签名；缺失或非法值拒绝执行。
每次仍必须出现匹配的请求签名及实际 tiling 记录，不能仅以缓存命中证明参数已生效。
缓存键包含源码包、适配代码、SoC、CANN 路径/版本记录、legacy 库和编译工具指纹，不包含 shape 或搜索参数。
命中前验证 OPP 和 legacy 库哈希，文件锁防止并发重复构建，失败构建不发布。缓存目录和生成工程保留稳定绝对路径。
修改 C++ 策略逻辑需要重新构建一次；调整现有参数不需要。

`--build-cache /home/ma-user/work/paretopilot-cache` 指定持久目录，`--no-build-cache` 恢复每参数构建。
JSON 配置对应 implementation.cache_dir、implementation.build_cache。日志打印 Source cache hit/miss，
selection.cache_hit/cache_key 记录包的来源；同一运行中的相同参数复测另记录 reused_build.json。
冷构建日志复制到试验的 cache-build-logs，原始工程在缓存目录。
每次仍重新生成 OM、执行正确性检查、预热并采样，各 ATC 使用独立工作/缓存目录。

## 测试命令

```bash
cd /home/ma-user/work/ParetoPilot
git pull --ff-only origin main
/home/ma-user/anaconda3/envs/PyTorch-2.6.0/bin/python3 scripts/test_910b.py \
  --cann-path /usr/local/Ascend/cann-8.5.0 \
  --soc Ascend910B2 --device 0 --deps check \
  --operator matmul --matmul-implementation batch-v3 --matmul-case all \
  --scope smoke --timeout 3600
```

smoke 检查二维、共享权重和独立 batch 三个用例，每个只执行一次默认 V3 的正确性与 profiling，保留 Roofline。
该路径跳过 V2 参照、候选、确认轮次；仅选 MatMul 的 smoke 跳过单元测试和模拟闭环。不能据此声明调优收益。
summary.scope=source_smoke，任务时间看 source_baseline.duration_us；没有候选数或加速比。
改成 `--scope full` 执行完整搜索；仅测一个用例可用 `--matmul-case 2d/shared_rhs/batched_rhs`，
`3d` 选择两个三维用例。无需 sudo、Agent 接口或安装新的依赖。

只生成一个用例的 V3 源码基线和 V2 参照，可在加载 CANN 环境后执行：

```bash
python3 -m paretopilot baseline --config examples/ascend910b_batchmatmul.json --output runs/bmmv3-baseline
```

该命令的 baseline_us 指 V3 默认源码，另给出 stock_reference_us；目录中 baseline_summary.json
是默认 V3，stock_reference.json 是 V2。完整搜索 summary.json 中 baseline 字段为兼容保留的 V2，
primary_baseline 字段明确标记 source_baseline。

当前本机只有 CPU 测试与 mock 验证，不能证明 910B2 上编译和数值路径已验收；上板结果以各用例
ATC/构建日志、correctness.json、profiling CSV 与有效 tiling 记录为准。
