# M001：Table IV 60M Dense Muon

## 2026-09-25：配置与入口检查

- 目的：建立与 GreedyLoRE + Muon 仅压缩方式不同的 Dense Muon 对照。
- 实验编号：`CM001-m001-dense-muon-llama60m-c4-dense-bf16-s1243`；BF16 正式训练尚未启动。
- 代码和配置：`c4/run_llama_pretraining.py`、`c4/configs/llama_60m_table_iv.json`、`c4/scripts/run_table_iv_60m_muon.bash`。
- 数据：`/dev/shm/wyr_tmp/c4` 下的本地英文 C4 train/validation shards；已验证能读取一条样本。输出路径为 `outputs/<实验编号>/`。
- 论文参照：Table IV 的 60M 列为 `dmodel=256`、约 1.1B 有效训练 token；附录 F 为 10,000 更新步、warmup 1,000 步、global batch 512、序列长 256、cosine 到峰值学习率的 10%、weight decay 0、梯度裁剪 1.0。论文没有给出完整 60M 架构或 Muon 学习率。
- 模型架构推定：hidden 256、8 heads、56 layers、FFN 688，实际 60,682,496 参数；不是已核实的作者原始配置。
- Muon 起始配置：matrix LR 0.02、momentum 0.95、spectral-norm LR scaling；AdamW fallback LR 0.001、betas (0.9, 0.999)。采用 4 GPU、每卡 microbatch 128、累积 1、activation checkpointing、BF16、seed 1243。Muon 参数不来自论文，和 M002 保持一致。
- 用户指定 W&B online，且不保存 checkpoint。脚本的 `save_every=20000` 大于 10,000 步预算；`save_dir` 沿用入口默认值但不会写入。单卡合成数据 batch 128 的完整前向、反向、Muon 更新峰值显存为 14.12 GiB。
- 验证：`PYTHONPATH=. .venv/bin/python -m unittest discover -s tests -v` 16 项通过；运行脚本 `--dry-run` 可生成完整命令。正式指标为最低及最终 validation PPL。
- 下一步：跟踪正式训练结果；本入口不保存可续跑的完整 checkpoint，异常退出需视为新 run。

## 2026-09-25：FP32 启动失败与 BF16 更正

- 首次启动误用 FP32，目录 `outputs/CM001-m001-dense-muon-llama60m-c4-dense-s1243/`；在第 0 步之前因当前 Transformers 中 `model.generation_config is None` 而退出，没有训练指标，CM002/CM003 未启动。
- 按用户更正，将三组改为 BF16；入口改为设置模型 `config.pad_token_id`，仅当 generation config 存在时再同步它。失败日志保留，不并入 BF16 结果。

## 2026-09-25：BF16 正式训练启动

- 启动命令：`tmux new-session -d -s greedylore_tableiv_60m_bf16 'cd /home/wyr/greedy_lore && bash c4/scripts/run_table_iv_60m_muon.bash dense && bash c4/scripts/run_table_iv_60m_muon.bash r32 && bash c4/scripts/run_table_iv_60m_muon.bash r128'`。
- 运行环境：GPU 3、4、5、6；W&B online；无 checkpoint。CM001 的 W&B ID 是 `b35vplch`，本地日志在 `outputs/CM001-m001-dense-muon-llama60m-c4-dense-bf16-s1243/train.log`。
- 启动检查：四个 rank 已完成进程组初始化；训练完成第 1 次更新并进入首轮 validation。当前无可报告的最终结果。

## 2026-09-25：数据耗尽与退出故障

- CM001 在本地 30 个 C4 train 分片耗尽时只达到 5,220 更新步，未达到原定 10,000 步；这次运行是部分预算，不能当作 Table IV 的完整结果。该 run 的 W&B ID 为 `b35vplch`，最终 validation PPL 约 42.57，仅作故障排查记录。
- 入口随后调用当前 W&B 0.30.0 已不存在的 `run.get_url()`，报 `AttributeError`；串行任务因此没有启动 CM002/CM003。因为无 checkpoint，这次运行不能续跑。
- 按用户明确的 **1.1B token 位置（含 padding）** 预算，512×256×8,393 = 1,100,087,296，将四组计划统一改为 8,393 更新步。这个预算口径与之前按附录 F 设置的 10,000 步不同。原 30 个训练分片仍不够，新增数据或复用现有数据的选择待定。
- 修复训练入口：未达到目标步数即数据耗尽时抛错，防止写出貌似完成的结果；正常结束时使用 W&B `run.url`。19 项单元测试通过，脚本 dry run 通过。待数据方案确定后，CM001 从头启动，使用唯一的重试 run ID。

## 2026-09-25：纠正数据量判断与 DataLoader 双重切分

- 上一节“30 个 C4 分片不够”的判断已被推翻。30 个分片实际有 10,689,518 条记录，每个约 356,317 条。训练的 `PreprocessedIterableDataset` 对 DataLoader worker 再次使用 `islice`，但 Hugging Face 流式数据集已自行按 worker 分片；这使每个 worker 只读取被分配记录的约四分之一。16 条记录的复现测试在修复前只返回 4 条，修复后返回全部 16 条。
- 首个分片按每 356 条抽取一条、共 1,001 条，以训练所用 T5 tokenizer 估计：每分片约 1.91 亿原始 token；截断至 256 后约 6,969 万非 padding token；全部填充时约 9,122 万 token 位置。30 个分片的名义总容量约 27.37 亿 token 位置，超过 8,393 步的 11.00 亿预算，无需补充数据。抽样估计不是精确逐条 token 统计。
- 删除包装器内的第二次 worker 切分；20 项单元测试通过，`git diff --check` 通过。此前 CM001 仅 5,220 步的根因是代码丢样本，不是原始分片不足。
- 已重新启动四组串行 tmux 会话 `greedylore_tableiv_60m_bf16`，顺序为 Dense Muon、GreedyLoRE r32、GreedyLoRE r128、Dense AdamW；每组都从头训练。Dense Muon 重试 ID 为 `CM001-m001-dense-muon-llama60m-c4-dense-bf16-s1243-rerun1`，其余三组沿用尚未启动过的 CM002–CM004 ID。已确认 torchrun 与四个训练 rank 启动；目前尚无完成结果。
- 进一步以 30 shard、4 rank、每 rank 4 worker 的小型回归测试确认 120 条记录全部恰好读取一次。Dense Muon 重试已完成首个更新，正在执行第 1 步 validation。

## 2026-09-25：130M Dense Muon 启动

- 实验编号：`CM005-m001-dense-muon-llama130m-c4-dense-bf16-s1243`，使用仓库 `c4/configs/llama_130m.json`（hidden 768，约 134.1M 参数），与后续三组固定同一架构。
- 训练：本地 C4、4 GPU（3–6）、每卡 batch 128、GA=1、序列长 256、20,000 更新步（2,621,440,000 个含 padding 的名义 token 位置）、warmup 2,000、cosine 到 10%、BF16、activation checkpointing、seed 1243。Muon matrix LR 0.02、momentum 0.95、spectral-norm scaling；scalar AdamW LR 0.001。梯度裁剪 1.0，weight decay 0。
- 用户授权 W&B online、无 checkpoint；`save_every=40000` 高于训练预算。运行入口 `c4/scripts/run_table_iv_130m.bash`，串行会话 `greedylore_tableiv_130m_bf16`，输出 `outputs/<实验编号>/`。已确认 torchrun 及四个 rank 启动；尚无结果。
- 启动前检查：30 个本地 shard 有 10,689,518 条记录，20,000 步需要名义上 10,240,000 条；脚本静态检查及 21 项单元测试通过。数据余量约 4.4%，训练入口会在提前耗尽时明确报错。
- 启动后检查：CM005 的 W&B ID 为 `op0yd9v0`，日志显示 `Syncing run`；训练已完成第 1 次参数更新并进入第 1 步 validation。GPU 3–6 各使用约 11 GiB；输出目录下尚无 checkpoint。

## 2026-09-26：Table III GLUE Dense Muon 对照排队

- 实验编号：`CM009`、`CM012`、`CM015`、`CM018`、`CM021`、`CM024`、`CM027`、`CM030`，依次对应 SST-2、CoLA、MRPC、STS-B、RTE、QNLI、QQP、MNLI；完整运行目录名含任务和 seed。
- 与同任务的 M002 rank 8/16 组使用相同 RoBERTa-base、GLUE 本地 Arrow 数据、seed 1243、FP32、4 GPU、每卡 batch 4、10 epoch、序列长 256、cosine 到 0、LR warmup 10%、weight decay 0。矩阵 Muon LR `2e-4`、momentum 0.95、spectral-norm scaling；AdamW fallback LR `5e-5`。学习率未 sweep。
- 运行入口为 `glue/scripts/run_table_iii_muon.bash`；先等待 GPU 0/1/2/7 的 GPU 利用率均低于 20%，不检查显存利用率，再串行执行 24 个任务。结果口径为 validation，不是论文的 test。
- 用户进一步要求轮询时避免前序任务训练间隙的短暂低利用率导致抢占。启动器现先等待前序 `cm008-table2-randk-topk-muon-gpu0-1-2-7` tmux 会话结束，再要求四张 GPU 连续 3 次（间隔 30 秒）利用率均低于 20%；显存利用率不作为门槛。启动器会话 `greedylore_tableiii_muon`，日志 `outputs/table_iii_muon_launcher.log`。截至启动检查仍在等待前序会话，尚未开始 CM009。

## 2026-09-26：Table III 四卡直接短试跑及调度修复

- 按用户更正暂停正式启动器，直接在 GPU 0/1/2/7 执行 SST-2 Dense Muon 4 个更新步（`--max_train_steps 4`，每卡 batch 4），成功完成 validation；结果 `outputs/smoke-direct-tableiii-dense/all_results.json`，accuracy 仅约 0.491，不作为正式性能指标。
- 短跑暴露 GLUE 入口的 LR 调度器在 4 卡下每个 optimizer step 推进 4 次，记录为 0、`2e-4`、0、`2e-4`。设置 `Accelerator(step_scheduler_with_optimizer=False)` 后，同样 4 步的 LR 依次为约 `1.707e-4`、`1e-4`、`2.929e-5`、0；验证产物 `outputs/smoke-direct-tableiii-dense-scheduler-fix/`。
- 同时将总更新步和 warmup 的计算移到 Accelerate 数据切分之后。用本地 SST-2 train 的 64 条临时子集做 4 卡、1 epoch、4 更新步检查，日志 `outputs/smoke-direct-tableiii-scheduler-auto.log` 显示自动预算为 4，25% warmup 后 LR 为 `2e-4`、`1.5e-4`、`5e-5`、0。临时子集仅用于调度检查，正式运行仍使用完整本地 GLUE 数据。
- 静态检查、24 条正式命令 dry-run、24 项单元测试通过后，于 09:14:58 重新启动 `greedylore_tableiii_muon`；启动检查仍在等待前序 CM008 tmux 会话退出，正式训练尚未开始。
- 用户更正启动条件：不等待 CM008 会话。正式启动器仅轮询 GPU 0/1/2/7 的 `utilization.gpu`，每 300 秒采样一次，连续两次都低于 20% 即开始；显存利用率不参与判断。旧启动器已停用，并按新条件重启；此前“必须等 CM008 退出”不再适用。

## 2026-09-26 12:00：Table III 改为 W&B online 后从头重跑

- 用户要求立即停止原本不带 W&B 的正式训练，禁止事后上传，并且不保存 checkpoint。已中断 `greedylore_tableiii_muon` 原会话；旧 CM009 已完成、CM010 在约 15,453/42,100 更新步中断，这些结果不纳入本次 W&B online 正式比较。
- 更新 GLUE 入口：W&B run 名使用 `--wandb_run_name`，正常结束时读取当前 W&B `run.url`；启动器设置 `WANDB_MODE=online`、`--with_tracking --report_to wandb`。各任务项目名为 `glue_no_trainer_<task>_greedylore`，同一项目包含 Dense/r8/r16；run 名与本地目录同为原 CM 编号加 `-rerun1`。没有设置 `--checkpointing_steps` 或 `--resume_from_checkpoint`，因此不保存训练 checkpoint。
- 删除未使用的事后上传脚本及测试。24 条命令 dry-run 均含 online tracking 参数且无 checkpoint 参数；GLUE CLI 与本地数据测试 3 项通过。12:00:44 重新创建 `greedylore_tableiii_muon` 会话，按 GPU 0/1/2/7 每 300 秒连续两次低于 20% 的条件等待；12:00:45 首次检测四卡均为 0%，计数 1/2。新训练尚未开始。
- 12:05:45 第二次检测四卡均为 0%，随即从头启动 CM009 `-rerun1`；12:05:59 W&B 日志显示 `Syncing run`，run ID `g5iydejl`，链接 https://wandb.ai/yiran0834-huazhong-university-of-science-and-technology/glue_no_trainer_sst2_greedylore/runs/g5iydejl 。12:06:04 已完成前 13 个更新步；新输出目录只有 `train.log`，没有 checkpoint。

## 2026-09-29：完成结果核对

- 以各 run 的 `outputs/<完整 run ID>/all_results.json` 和 `train.log` 核对：CM001 `-rerun1` 完成修订后的 8,393 步，final C4 validation loss/PPL 为 3.61430 / 37.1255；CM005 完成 20,000 步，为 3.29619 / 27.0095。两组日志均有 `Script finished successfully`。早期只到 5,220 步的 CM001 仍是失败记录，不并入正式结果。
- GLUE Dense Muon 的 CM009、CM012、CM015、CM018、CM021、CM024、CM027、CM030 均以 `-rerun1` 完成第 10 个 epoch，并产生机器可读结果。最终 validation 指标和对应源字段已整理在 `docs/results.md`；MNLI matched 的最终指标来自 epoch 9 日志，JSON `eval_accuracy` 对应 mismatched。
- 上述结果均只有 seed 1243，属于初步结果；C4 60M 的 8,393 步与论文附录的 10,000 步不同，GLUE 是 validation 而非论文 test。
