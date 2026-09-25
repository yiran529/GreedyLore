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
