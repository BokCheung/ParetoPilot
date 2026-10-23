# 定位安装包中的 BatchMatMulV2

在已通过 smoke 的测试机执行。脚本读取 CANN 安装目录，复用既有内置 baseline 的 ATC 命令；诊断日志、缓存和新 OM 写入独立的 `runs/inspect-bmmv2-*`。它不安装依赖、不修改系统 OPP、不执行性能测试。保留当前环境中的 Python 依赖即可。

```bash
cd /home/ma-user/work/ParetoPilot
git pull --ff-only origin main
python3 scripts/inspect_cann_operator.py \
  --cann-path /usr/local/Ascend/cann-8.5.0 \
  --baseline-dir runs/910b-r4yr55e1/matmul-smoke/baseline \
  --timeout 3600
```

`--baseline-dir` 必须是内置 `baseline/`，不能是 `source_baseline/` 或候选目录。若之前的 run 已删除，先重新运行 MatMul smoke，再换成新目录。SoC、输入 shape 和 ONNX 从该目录的 `atc.command.json` 读取，避免手工重写后偏离原来的测试。ATC 只重编译，不执行 OM，因此不需要额外指定设备 ID。

可加 `--scan-only` 只查安装目录；`--no-strace` 只用 CANN 日志。默认检测已安装的 `strace` 并先测试可用性，缺少或不允许跟踪时自动降级，不调用 sudo、不自动安装。没有文件跟踪证据时，不能仅凭文件存在判断实际加载路径。

## 查看输出

脚本最后打印 `report.txt` 的绝对路径。优先提供以下三个文件内容：

1. `report.txt`：CANN/ATC 路径和版本、原模型哈希、注册文件、候选源码及跟踪状态。
2. `registration-and-sources.json`：`BatchMatMulV2` 的注册条目，包括可用的 `opFile`、`opInterface`、输入格式/类型、属性；候选 Python 文件的参数、导入和编译/tiling 相关行。
3. `log-matches.txt`：ATC、Host 日志、编译统计和 dump 图中的相关片段。最多摘录 1000 条，完整日志仍保留。

进一步查看：

- `trace-files.log`：`strace -f` 跟踪 ATC 及子进程的文件打开和编译器启动。
- `opened-files.json`：成功打开的文件/执行程序；排除 `ENOENT` 等失败探测。这里包含进程级依赖，不能把所有库都归属于目标算子。
- `library-dependencies.txt`：如有 `readelf`，对实际打开的相关 Host/tiling 库提取 `NEEDED`、`RPATH`、`RUNPATH`，不执行库文件。
- `collect/`、`logs/`、`graphs/`：完整编译记录、Host 日志及图结构。若本版本生成 `op_compile_stats.log`，其中的在线编译、缓存复用、二进制复用信息有助于解释入口选择。

## 如何判断可修改范围

| 证据 | 能确认什么 | 还需确认什么 |
| --- | --- | --- |
| 注册条目中的 opFile/opInterface | 候选编译模块及函数名 | 用日志和文件跟踪确认实际选择；系统 vendors 优先级也会影响加载 |
| Python 入口及 import | 包装函数参数、依赖模块、调度/编译调用位置 | 依赖是否继续转入不可修改的 `.so` 或预编译 kernel |
| 完整 Python DSL / C++ 内核与 tiling 源码 | 存在源码适配的可能 | 编译工具、依赖、许可证及版本能否配齐；复制到隔离目录后先复现原性能 |
| 只有 `.so`、`.o`、二进制与 Python 包装 | 能调用该实现 | 不能据此认为已取得完整内核源码，只能使用已公开并验证生效的参数接口 |
| 注册属性/函数参数中的 impl_mode、转置、格式等 | 入口接受这些字段 | 哪些字段保持当前 FP16 语义且实际影响性能；不能把所有字段直接当调优参数 |

应先把“注册配置 → 实际入口 → 调度/tiling 依赖 → 内核/二进制”串起来，再决定源码改造或内置调优路线。单独发现 `batch_matmul_v2.py`、单独看 CSV 的 Op Type、或把 V3 改名为 V2，都不能证明重建了当前内置实现。

新的诊断缓存不保证强制源码编译：安装包仍可能提供预编译二进制。打开文件只证明进程访问过它；判断某个分支或参数是否生效，还需要对应日志和后续隔离实验。诊断模式的耗时不参与性能比较。

环境变量依据：[CANN 8.5.0 图结构导出](https://www.hiascend.com/document/detail/en/canncommercial/850/maintenref/envvar/envref_07_0011.html)、[CANN 8.5.0 日志级别](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/850/maintenref/envvar/envref_07_0122.html)。脚本只对子进程设置调试环境，退出后不改变调用终端的环境。
