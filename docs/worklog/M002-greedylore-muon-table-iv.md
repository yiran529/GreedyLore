# M002：Table IV 60M GreedyLoRE + Muon

## 2026-09-25：通信实现与配置检查

- 目的：在 M001 的相同 Muon、模型、数据、seed 和预算下，仅改变 DDP 梯度通信，比较 rank 32 与 rank 128 的 validation PPL 和通信代价。
- 实验编号：`CM002-m002-greedylore-muon-llama60m-c4-r32-bf16-s1243` 与 `CM003-m002-greedylore-muon-llama60m-c4-r128-bf16-s1243`；均尚未启动正式训练。
- 代码和配置：`comm_hooks/subspace_hook.py`、`comm_hooks/utils.py`、`c4/configs/llama_60m_table_iv.json`、`c4/scripts/run_table_iv_60m_muon.bash`。
- 方法：真实通信路径为 `top_subspace`。压缩对象是 DDP bucket 中符合压缩门槛的参数梯度，发生在 Muon 动量和正交化前，属于近似 Muon；hook 可能同时覆盖 Muon 矩阵和 AdamW fallback 的矩阵梯度。embedding 与 lm_head 按现有规则不压缩。压缩 rank 为 32 或 128，EF14、前 1,000 个通信迭代 dense、投影更新间隔 200。
- 修复：原实现非 SVD 更新步的方向评分函数只有 `pass`。单个二维矩阵在第二步稳定复现 `TypeError: 'NoneType' object is not subscriptable`；现在对每个基向量使用独立高斯探针，平均各 rank 的有符号分数，再平方选方向。`lore` 仅复用固定 SVD 投影，`fake_top_subspace` 做完整梯度 AllReduce，不用于正式 GreedyLoRE 通信实验。
- rank 128 门槛：`hidden_size=256` 时默认最小压缩率 2.0 会跳过临界的 256 行/列矩阵；两组均显式使用 1.0，使实际被压缩的矩阵可进入 hook。运行时仍需报告实际压缩覆盖率。
- 训练设置随 M001 使用每卡 batch 128、GA=1、activation checkpointing、BF16；W&B online、不保存 checkpoint。
- 验证：单矩阵回归测试、双 rank Gloo 与双 rank NCCL/CUDA 小测试通过。正式训练未启动，当前无 PPL 或通信节省结论。
- 下一步：CM001 完成后，同一 `tmux` 脚本将串行启动 CM002、CM003。已用双 rank NCCL/CUDA 验证二维矩阵的投影更新和间隔压缩步一致性；正式结果仍待运行完成后检查。

## 2026-09-25：预算调整与串行任务中断

- CM001 在 5,220 步耗尽当前训练数据，随后因 W&B URL 接口报错退出，CM002/CM003 未启动。
- 用户将四组共同训练量明确为约 1.1B token 位置（含 padding）；脚本改为 8,393 步。数据补足方式待定，确定后再启动。

## 2026-09-25：数据量纠正

- 查明 C4 包装器与 Hugging Face 流式数据集对 DataLoader worker 重复分片，导致前次只使用约四分之一的记录。已修复并以多 worker 单元测试确认无丢样本。现有 30 个分片足够 8,393 步，无需下载或循环数据。

## 2026-09-25：130M 两组串行排队

- 实验编号：`CM006-m002-greedylore-muon-llama130m-c4-r32-bf16-s1243`、`CM007-m002-greedylore-muon-llama130m-c4-r256-bf16-s1243`。
- 与 CM005 完全共用模型 hidden 768、20,000 步、Muon 超参数、C4、seed、BF16、每卡 batch 128、GA=1、W&B online、无 checkpoint；仅改变通信为 `top_subspace`，rank 分别为 32 和 256。EF14、从第 1,000 步压缩、投影更新间隔 200、最小压缩率 1.0。
- 两组在 `greedylore_tableiv_130m_bf16` 内排在 CM005 后；若前序失败，`&&` 阻止后续运行。论文 Table IV 把 130M 的 rank 32 栏写为 `32/512`，与本仓库 130M hidden 768 不一致；此处固定 hidden 768，以保持方法间可比。

## 2026-09-26：Table III GLUE GreedyLoRE + Muon 两秩排队

- 与 M001 GLUE Dense Muon 共用模型、数据、Muon 超参数、batch、seed、训练预算和评估口径，仅改为 `--compressor top_subspace`、`--compress_rank 8/16`、`--use_error_feedback ef14`、`--start_compress_iter 1000`、`--update_proj_gap 200`。rank 8 为 `CM010/013/016/019/022/025/028/031`；rank 16 为 `CM011/014/017/020/023/026/029/032`，任务顺序见 M001。
- 论文附录对 rank 16 的 QNLI 使用 global batch 32；本比较统一 global batch 16，避免 batch 与 rank 同时变化。压缩作用于 Muon 动量和正交化前的 DDP 梯度；hook 可能覆盖 AdamW fallback 矩阵梯度，因此属于近似 Muon。
- 本地数据目录 `/home/wyr/.cache/huggingface/datasets/glue/` 已核对八任务 train/validation Arrow 文件，MNLI 同时具备 matched/mismatched validation。GLUE 入口增加本地 Arrow 读取，避免当前 `load_dataset("nyu-mll/glue")` 离线不命中缓存；未提供额外 test 数据，最终只报告 validation 指标。

## 2026-09-26：Table III rank 8/16 直接短试跑

- 按用户要求，直接在 GPU 0/1/2/7 上对 SST-2 的 `top_subspace` rank 8 和 rank 16 各执行 4 个更新步；短跑临时改为第 1 次通信迭代开始压缩、投影更新间隔 2，以覆盖投影更新和非更新的压缩路径。正式配置仍为第 1000 次开始、间隔 200。
- 两组均完成第 4 步及 validation，日志与结果分别为 `outputs/smoke-direct-tableiii-r8.log`、`outputs/smoke-direct-tableiii-r8/all_results.json` 和 `outputs/smoke-direct-tableiii-r16.log`、`outputs/smoke-direct-tableiii-r16/all_results.json`；未观察到 collective 错误或 NaN。两组均在第 1、3 次迭代记录投影更新。短跑指标不用于方法比较。

## 2026-09-26：130M 秩与压缩起点的独立尝试排队

- 目的：分别检验提高秩与延后开始压缩能否缩小 130M GreedyLoRE + Muon 对 Dense Muon 的 validation PPL 差距；两个实验各只改变一个变量。
- `CM033-m002-greedylore-muon-llama130m-c4-r64-bf16-s1243`：相对 CM006 仅将 `compress_rank` 从 32 提到 64，`start_compress_iter=1000`。
- `CM034-m002-greedylore-muon-llama130m-c4-r32_sci2000-bf16-s1243`：相对 CM006 保持 rank 32，仅将 `start_compress_iter` 从 1000 提到 2000。
- 其余沿用 `c4/scripts/run_table_iv_130m.bash`：hidden 768、20,000 步、4 GPU、global batch 512、每卡 batch 128、GA=1、BF16、Muon matrix/scalar LR 0.02/0.001、EF14、`update_proj_gap=200`、`min_compression_rate=1.0`、seed 1243、W&B online 项目 `GreedyLore-TableIV`，不保存 checkpoint。
- 使用 `c4/scripts/queue_table_iv_130m_followups.bash` 后台轮询：先等原 `greedylore_tableiv_130m_bf16` 会话结束，并核实 CM007、CM008 都有 `update_step=20000` 的结果；之后要求 GPU 3–6 连续三次（间隔 30 秒）利用率不超过 10%、显存占用不超过 1,500 MiB，再串行启动 CM033、CM034。每项启动前重新等待；第一项失败也记录状态并尝试第二项。队列状态在 `outputs/CM033-CM034-greedylore-130m-followups/status.tsv`，各 run 的训练日志在自身输出目录。
- 静态验证：两项 dry-run 均确认仅预期参数和 run 路径变化，W&B 项目、20,000 步、BF16、GA=1、无 checkpoint 设置保持一致；脚本通过 `bash -n`，输出目录尚不存在。当前原 130M 队列仍在跑 CM007，因此新队列不立即占用 GPU。
- 已创建 tmux 会话 `greedylore_tableiv_130m_followups`；`status.tsv` 显示 `waiting`，同时原 `greedylore_tableiv_130m_bf16` 会话仍在运行。当前尚未启动 CM033/CM034 的训练进程。

## 2026-09-26：后续队列未启动的原因与修正

- 原 CM007 在 20,000 步后正常写出 `all_results.json`，日志显示脚本成功结束；原 130M tmux 会话随后消失，但 CM008 没有输出目录或结果。会话为何在 CM008 启动前中断，目前没有直接证据。
- 新队列首次运行于 01:58 在 `status.tsv` 写入 `blocked` 后退出：它原先要求 CM007、CM008 均已有完整结果，因此没有抢占 GPU；CM033/CM034 均未启动。保留该状态记录。
- 修正 `c4/scripts/queue_table_iv_130m_followups.bash`：原会话结束且 CM007 完成后，若 CM008 未完成，则先经 GPU 3–6 连续空闲轮询启动既有 `adam_dense` 配置；仅当 CM008 完成 20,000 步，才继续 CM033、CM034。任何 CM008 失败均记录并阻止后续运行。新队列以追加方式写状态，不覆盖首次阻塞记录。
- 首次重启时 GPU 3–6 显存均仅约 3 MiB，但 10% 利用率阈值受瞬时波动影响，连续三次空闲检查尚未通过；将阈值改为 50%，仍要求显存低于 1,500 MiB、连续三次检查，并重启本队列。此前状态记录保留。
- 用户随后明确要求直接占用当前空闲的 GPU 3–6。确认四张卡显存均约 3 MiB 后，停止本次仍处于等待阶段的队列，并以 `--start-now` 重启同名 tmux 会话；09:06 已启动补跑 CM008。CM033、CM034 保持串行排在 CM008 完成校验后，两个尝试仍分别只改变秩或压缩起点。状态文件保留两次等待及首次阻塞记录。

## 2026-09-26 12:00：Table III GreedyLoRE 改为 W&B online

- 原 Table III CM010 rank 8 SST-2 在约 15,453/42,100 更新步中断，原批次没有 W&B online，不纳入正式比较。rank 16 尚未启动。
- 新的 rank 8/16 运行沿用 CM 编号、目录与 run 名追加 `-rerun1`，从头训练；W&B online 项目按任务命名为 `glue_no_trainer_<task>_greedylore`，不设置 checkpoint。与 Dense Muon 共用 GPU 0/1/2/7 空闲门槛、Muon 参数和训练预算，串行排在同一任务 Dense 后。

## 2026-09-29：完成结果核对

- C4 60M 的 CM002 r32、CM003 r128 均完成修订后的 8,393 步，final validation PPL 分别为 40.5794、37.4963；同预算 Dense Muon CM001 `-rerun1` 为 37.1255。C4 130M 的 CM006 r32、CM007 r256、CM033 r64、CM034 r32 且延后到第 2,000 步压缩，均完成 20,000 步，final PPL 分别为 36.4617、28.8555、33.5800、34.3349；同预算 Dense Muon CM005 为 27.0095。数值来自各 run 的 `all_results.json`，日志均有成功结束标记。
- 后续队列状态文件记录 CM008、CM033、CM034 均以 exit code 0 结束；各自的 `all_results.json` 又确认达到 20,000 步。CM033/CM034 是观察 CM006 后设计的单因素探索，不能把同一 validation 上的改善视作独立验证。
- GLUE rank 8 的 CM010/013/016/019/022/025/028/031 与 rank 16 的 CM011/014/017/020/023/026/029/032 均以 `-rerun1` 完成第 10 个 epoch。最终 validation 指标及源字段见 `docs/results.md`；原无 W&B 的 CM010 中断尝试继续排除在正式比较之外。所有结果只有一个 seed。

## 2026-09-30：130M r64 Muon LR 宽范围筛选

- 目的：在固定 GreedyLoRE r64、EF14 和通信设置时，分别筛选 Muon 矩阵 LR 与 scalar AdamW LR；门槛取既有 CM033 在第 5,000 步的 validation loss `3.743565253792393`，候选必须严格更低才继续到 20,000 步。CM033 与本轮使用的 C4 路径不同，这一门槛仅用于探索性筛选。
- 实验编号：CM036 (0.004/0.001)，CM037 (0.1/0.001)，CM038 (0.02/0.0002)，CM039 (0.02/0.005)；括号内为矩阵 LR/scalar LR。每项从头训练，seed 1243。
- 其余参数沿用 `c4/scripts/run_table_iv_130m.bash r64`：LLaMA 130M、C4、4 GPU、BF16、序列长 256、全局 batch 512、20,000 步 cosine 调度、warmup 2,000、rank 64、EF14、压缩起点 1,000、投影间隔 200、无 checkpoint、W&B online。数据路径固定为 `c4/c4_en`，GPU 4–7。
- 队列 `c4/scripts/sweep_table_iv_130m_lr.py` 读取每个 run 的 `train.log` 中 `Eval loss at step 5000`；候选 loss 等于或高于 CM033 时终止该候选进程组，低于则在原 20,000 步 LR 调度下继续同一次运行。状态记录于 `outputs/CM036-CM039-r64-lr-sweep/status.tsv`。不修改训练入口或优化器。
- 启动前核对：50 个训练分片和 8 个验证分片可由本地加载器读取；候选 dry-run 均保持 r64、EF14、20,000 步 cosine、warmup 2,000 等既有参数；相关单元测试、脚本语法与 `git diff --check` 通过。结果和运行状态以输出目录中的日志、`all_results.json` 和队列状态为准。
- 已启动 tmux 会话 `greedylore_130m_r64_lr_sweep`；队列状态记录旧 CM033 为 `reference`，当前 GPU 4–7 被其他任务占用，队列在等待连续空闲检查后启动 CM036。尚无本轮候选结果。
- 按用户后续要求，将正式队列的资源门槛改为：检查全部 GPU，每 5 分钟取样一次，连续两次取样中同样至少 4 张卡均满足显存 ≤1,500 MiB、利用率 ≤10% 才启动，实际选用的 4 张卡写入状态文件。等待控制器已在无候选训练时重启；CM036–CM039 配置和第 5,000 步 loss gate 不变。10 项相关单元测试通过；启动检查确认新控制器仍在等待，当前全部 GPU 被占用。

## 2026-10-01：终止 r64 LR sweep，改为 Dense 选 LR 后比较 r32/r256

- 用户观察降低 matrix LR 后的趋势，要求终止原 r64 sweep。CM036 在约第 12,000 步停止，未完成 20,000 步；CM037–CM039 未启动。CM036 的第 5,000 步与第 12,000 步 validation loss 分别为 3.48792、3.23633，仅为中断运行的观察值；保留原日志/W&B，不作为最终结果。状态见 `outputs/CM036-CM039-r64-lr-sweep/status.tsv`。
- 新实验 `CM042` r32 和 `CM043` r256 在 M001 的 CM040/CM041 完成后，使用最终 validation loss 更低的矩阵 LR。其余沿用 `c4/scripts/run_table_iv_130m.bash`、scalar LR 0.001、EF14、压缩从第 1,000 次通信迭代开始、投影间隔 200、20,000 步、C4 `c4/c4_en`、seed 1243、W&B online、无 checkpoint。完整 run ID 将含选定 LR。
- 串行队列脚本 `c4/scripts/queue_table_iv_130m_muon_lr.py`；每组启动前等待连续两次 5 分钟间隔的任意同一组 4 张 GPU 空闲。首组 Dense Muon 按用户单独授权直接用 GPU 0–3。队列状态在 `outputs/CM040-CM043-muon-matrix-lr/status.tsv`。

## 2026-10-02：r64 中断状态及 r32/r256 完成结果核对

- 目的与配置：记录最近学习率探索结果。CM036 沿用 r64 sweep 配置；CM042/CM043 使用 Dense 两候选选出的矩阵 LR 0.01、scalar LR 0.001，其他配置沿用上一条记录。实际 GPU 均为 0–3 的 4 张 RTX 4090。
- 中断尝试：CM036 的日志 `Eval loss at step 5000` / `12000` 分别为 3.48792 / 3.23633；虽通过旧 CM033 的 5,000 步筛选门槛，但用户改换实验后终止。没有最终结果文件或成功结束标记；CM037–CM039 取消且未启动。状态见 `outputs/CM036-CM039-r64-lr-sweep/status.tsv`，指标见 `outputs/CM036-m002-greedylore-muon-llama130m-c4-r64-lr0p004-slr0p001-bf16-s1243/train.log`。这两项是中间 validation loss，不作为 final。
- 完成结果：`CM042-m002-greedylore-muon-llama130m-c4-r32-lr0p01-bf16-s1243`、`CM043-m002-greedylore-muon-llama130m-c4-r256-lr0p01-bf16-s1243` 均完成 20,000 步；final validation loss/PPL 分别为 3.20127 / 24.5636、3.13895 / 23.0797。
- 验证：核对 `outputs/<上述完整 run ID>/all_results.json` 的 `update_step`、`final_eval_loss`、`final_eval_ppl`，日志的最终 loss 和成功结束标记，以及 `outputs/CM040-CM043-muon-matrix-lr/status.tsv` 的 completed/queue finished。运行配置确认同为 `c4/c4_en`、rank 32/256、EF14 与所选 LR。
- 结论：同设置 Dense CM040 的 PPL 为 22.9218；r32 高 1.6418（7.16%），r256 高 0.1579（0.69%）。r256 在本轮更接近 Dense。该比较只有 seed 1243 一次，并在同一 validation 上选择学习率，属于初步探索；不能将跨数据路径的旧 CM006/CM007 差异单独归因于 LR，也未核算总通信收益。
- 下一步：固定所选配置，用预先规定的多 seed 确认；完整比较摘要见 `docs/results.md`。
