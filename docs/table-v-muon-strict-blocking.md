# GreedyLoRE Muon 严格阻塞通信测速（2026-10-03）

## 目的与口径

本实验比较 Dense Muon 与 GreedyLoRE + Muon 在通信完全阻塞时的耗时，用于观察取消
反向计算/通信重叠后，梯度压缩能否降低通信路径耗时。它是系统性能诊断，不测收敛，
也不把 hook 时间解释成纯 NCCL wire time：GreedyLoRE 的压缩、投影选择、collective、
解压和同步都计入 hook 时间。

严格阻塞只在独立入口 `c4/table_v_timing.py` 中包装现有 hook：每个 DDP hook 入口先
执行 CUDA synchronize，随后调用仓库原 hook、等待返回 Future 完成，再次 synchronize
后才允许 backward 继续。Dense 和 GreedyLoRE 使用同一包装器；不修改
`comm_hooks/subspace_hook.py`。入口同时记录每步 hook 总时间、实际 bucket 数与字节布局。

## 共同设置

- 本地英文 C4，LLaMA 60M/130M/350M/1B，序列长 256，seed 1243，GA=1。
- Muon matrix/scalar LR 0.01/0.001，momentum 0.95，spectral-norm scaling，weight
  decay 0，gradient clipping 1。
- GreedyLoRE 使用 `top_subspace`、rank 32、EF14、`start_compress_iter=100`、
  `update_proj_gap=200`、`min_compression_rate=1`、`beta_ef=0`、
  `error_inherit=0`。压缩发生在 Muon 动量与正交化之前，因此是近似 Muon。
- FP32，activation checkpointing 关闭，DDP `bucket_cap_mb=8192`，NCCL 默认 channel，
  P2P 关闭、SHM 开启。8192 MiB 大于四个模型各自的 FP32 梯度总量；结果仍以记录的
  实际 bucket 布局为准，不能仅依据 cap 推定为单 bucket。
- 预热 101 步；第 101 步建立首次投影。连续测量更新 102–501 共 400 步，其中更新
  301、501 为投影刷新。每个实验臂只运行一次。
- 主系统指标为最慢 rank 的连续 400 步墙钟时间除以 400；通信路径指标为各 rank
  400 个逐步 hook 总时间之和的最大值除以 400。两者分别保存为
  `mean_iteration_seconds` 和 `mean_blocking_hook_seconds`。
- 不评估、不保存 checkpoint、不连接 tracker。OOM、超时或其他失败保留日志并继续
  下一臂，不自动缩小 batch、模型或精度。

## 实验矩阵

每组运行四个模型和 Dense/GreedyLoRE 两臂，共 4×4×2=32 个实验，编号
CM134–CM165。

| 组 | GPU | 每卡 batch | 与论文配置的关系 |
| --- | ---: | ---: | --- |
| `paper-batch-ws4` | 4 | 60M/130M/350M 为 128；1B 为 64 | 保留论文工作负载的 world size、batch、模型和调度参照；严格阻塞、FP32、无 checkpointing 和大 bucket 是本实验控制项 |
| `b32-ws4` | 4 | 32 | 检验较大计算负载下的严格阻塞耗时 |
| `b1-ws4` | 4 | 1 | 降低每步计算量，观察 4 卡通信占比 |
| `b1-ws8` | 8 | 1 | 在低计算量下提高通信参与卡数 |

## 入口与产物

控制器为 `c4/scripts/run_table_v_muon_strict_blocking.py`，输出根目录为
`outputs/CM134-CM165-table-v-muon-strict-blocking/`。控制器串行执行 32 臂，保存
manifest、实际命令、NCCL 环境、逐臂日志、状态、原始 JSON、汇总表和配对 speedup。

```bash
PYTHONPATH=. .venv/bin/python c4/scripts/run_table_v_muon_strict_blocking.py --dry-run
PYTHONPATH=. .venv/bin/python c4/scripts/run_table_v_muon_strict_blocking.py --gpus 0,1,2,3,4,5,6,7
```

启动前的 8 卡 60M FP32/batch1 三步 smoke test 覆盖 Dense 预热、首次投影和一次压缩
测量步，结果位于 `/tmp/strict-blocking-smoke-20261003/`。该步实际记录 1 个 bucket，
大小 242,729,984 bytes；smoke 数值不用于方法比较。
