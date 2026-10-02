# GreedyLoRE + Muon 实验结果（2026-10-02 更新）

以下均为本仓库复现实验，**不是论文报告值**。每项只运行 seed 1243 一次，因此差异仅作初步观察。原始结果位于 `outputs/<完整 run ID>/all_results.json`，日志位于同目录的 `train.log`；完整 run ID 可由表中编号在 `outputs/` 下唯一定位。输出目录被 Git 忽略，本文保留可提交的结果摘要。

## C4 / LLaMA：Table IV 相关实验

### 核心设置与结果

本地 C4、4 GPU（3–6）、BF16、序列长 256、全局 batch 512、seed 1243、cosine 调度、weight decay 0，均未保存 checkpoint。60M 使用 `c4/scripts/run_table_iv_60m_muon.bash`，预算为修订后的 8,393 步、warmup 1,000；130M 使用 `c4/scripts/run_table_iv_130m.bash`，预算 20,000 步、warmup 2,000。两种模型规模不能直接比较。Muon 矩阵 LR 0.02、momentum 0.95、spectral-norm scaling，scalar AdamW LR 0.001；Dense AdamW LR 0.0025。压缩组使用 `top_subspace`、EF14、投影间隔 200；默认从第 1,000 步开始，CM034 改为第 2,000 步。

| 实验 | 模型 | 方法 | 完成更新步 | final validation loss | final validation PPL |
| --- | --- | --- | ---: | ---: | ---: |
| CM001 `-rerun1` | LLaMA 60M | Dense Muon | 8,393 | 3.61430 | 37.1255 |
| CM002 | LLaMA 60M | GreedyLoRE + Muon r32 | 8,393 | 3.70326 | 40.5794 |
| CM003 | LLaMA 60M | GreedyLoRE + Muon r128 | 8,393 | 3.62424 | 37.4963 |
| CM004 | LLaMA 60M | Dense AdamW | 8,393 | 3.50195 | 33.1799 |
| CM005 | LLaMA 130M | Dense Muon | 20,000 | 3.29619 | 27.0095 |
| CM006 | LLaMA 130M | GreedyLoRE + Muon r32 | 20,000 | 3.59626 | 36.4617 |
| CM007 | LLaMA 130M | GreedyLoRE + Muon r256 | 20,000 | 3.36230 | 28.8555 |
| CM008 | LLaMA 130M | Dense AdamW | 20,000 | 3.16086 | 23.5908 |
| CM033 | LLaMA 130M | GreedyLoRE + Muon r64 | 20,000 | 3.51393 | 33.5800 |
| CM034 | LLaMA 130M | GreedyLoRE + Muon r32，第 2,000 步开始压缩 | 20,000 | 3.53616 | 34.3349 |

表中数值直接来自各 run 的 `all_results.json` 字段 `update_step`、`final_eval_loss`、`final_eval_ppl`；入口在完成目标步数并再次评估后写入该文件，日志均有 `Script finished successfully`。60M 的 8,393 步约为 1.1B 个含 padding 的 token 位置，与论文附录的 10,000 步不同。最初的 CM001（无 `-rerun1`）只运行到 5,220 步，不纳入表格。CM008 是原队列未启动后补跑的同配置实验；CM033/CM034 是在查看 CM006 结果后设计的探索性尝试。

在相同 130M Muon 设置下，CM006、CM007、CM033、CM034 的 final PPL 分别比 Dense Muon CM005 高 9.4522、1.8460、6.5705、7.3254。CM034 比同为 r32 的 CM006 低 2.1267，但它使用同一 validation 数据进行探索，不是独立确认。Dense AdamW 使用不同优化器，不属于仅改变压缩方式的 Muon 对照。

## C4 / LLaMA 130M：最近的学习率选择与压缩比较（2026-10-02 核对）

### 实验矩阵与核心设置

CM040–CM043 均完成 20,000 更新步，模型配置为 `c4/configs/llama_130m.json`。本轮使用 C4 `c4/c4_en`；运行配置记录的数据路径与早期 CM005–CM008、CM033–CM034 的 `/dev/shm/wyr_tmp/c4` 不同，因此不将跨批次差异单独归因于学习率。所有指标均为本地复现的 **final validation**，不是论文结果。

共同设置：4 张 NVIDIA GeForce RTX 4090（GPU 0–3）、BF16、序列长 256、每卡 batch 128、全局 batch 512、GA=1、seed 1243；warmup 2,000 步、cosine 到峰值 LR 的 10%、weight decay 0、梯度裁剪 1.0。Muon momentum 0.95、spectral-norm scaling，scalar AdamW LR 0.001。压缩组采用 `top_subspace`、EF14、从第 1,000 次通信迭代开始压缩、投影间隔 200，秩分别为 32/256。运行入口为 `c4/scripts/run_table_iv_130m.bash`，队列为 `c4/scripts/queue_table_iv_130m_muon_lr.py`；未保存 checkpoint。

### 结果

| 实验 | 模型 / 方法 | 矩阵 LR | 完成更新步 | final validation loss | final validation PPL |
| --- | --- | ---: | ---: | ---: | ---: |
| CM040 | LLaMA 130M / Dense Muon | 0.01 | 20,000 | 3.13209 | 22.9218 |
| CM041 | LLaMA 130M / Dense Muon | 0.005 | 20,000 | 3.14294 | 23.1718 |
| CM042 | LLaMA 130M / GreedyLoRE + Muon r32 | 0.01 | 20,000 | 3.20127 | 24.5636 |
| CM043 | LLaMA 130M / GreedyLoRE + Muon r256 | 0.01 | 20,000 | 3.13895 | 23.0797 |

每个编号在 `outputs/` 下唯一对应一个完整 run ID。表中步数和指标来自该目录 `all_results.json` 的 `update_step`、`final_eval_loss`、`final_eval_ppl`，并与 `train.log` 的 `Final eval loss`、`Script finished successfully` 交叉核对。共同设置以本地运行配置 `wandb/run-*/files/config.yaml` 和启动脚本核对；实际 GPU 编号及完成状态来自 `outputs/CM040-CM043-muon-matrix-lr/status.tsv`，GPU 型号来自本地 `wandb-metadata.json`。

### 结论与未完成尝试

- 先比较 CM040/CM041 在第 20,000 步的 final validation loss，再将较低者的矩阵 LR 用于 CM042/CM043；实际选中 0.01。CM040 比 CM041 的 loss 低 0.01085、PPL 低 0.2500。这是两候选的 validation 选择结果，不代表全局最优 LR。
- 同为矩阵 LR 0.01 时，r32 相对 Dense 的 loss 高 0.06918、PPL 高 1.6418（7.16%）；r256 的 loss 高 0.00686、PPL 高 0.1579（0.69%）。本轮 r256 的验证指标更接近 Dense，尚不能据此判断总通信收益。
- 每组只有 seed 1243 一次运行（n=1），学习率选择与比较使用同一 validation，属于探索性结果；下一步需固定配置，用预先约定的多 seed 重复确认。
- 先前 r64 sweep 的 CM036（矩阵 LR 0.004 / scalar LR 0.001）第 5,000 步 validation loss 为 3.48792，低于筛选门槛 CM033 的 3.74357，继续运行后在第 12,000 步评估完被用户终止；该步 loss 为 3.23633。两项来源均为 `outputs/CM036-m002-greedylore-muon-llama130m-c4-r64-lr0p004-slr0p001-bf16-s1243/train.log` 的 `Eval loss at step <step>`，不是 final。该目录没有 `all_results.json` 或成功结束标记；状态文件 `outputs/CM036-CM039-r64-lr-sweep/status.tsv` 记为 `interrupted`。CM037–CM039 记为取消且未启动。CM033 与 CM036 的数据路径不同，筛选门槛只作探索参考。

## CIFAR / ResNet-18：Figure 3 协议下的 Muon 比较

### 实验矩阵

| 实验 | 数据集 | 模型 / 方法 | 完成预算 |
| --- | --- | --- | --- |
| CM044 | CIFAR-10 | ResNet-18 / Dense Muon | 40 epochs / 15,640 更新步 |
| CM045 | CIFAR-10 | ResNet-18 / GreedyLoRE + Muon r64 | 40 epochs / 15,640 更新步 |
| CM046 | CIFAR-100 | ResNet-18 / Dense Muon | 40 epochs / 15,640 更新步 |
| CM047 | CIFAR-100 | ResNet-18 / GreedyLoRE + Muon r64 | 40 epochs / 15,640 更新步 |

### 核心设置

每个数据集使用官方 50,000 张 train / 10,000 张 test。共同设置：4 张 RTX 4090、每卡 batch 32、全局 batch 128、seed 1243、FP32 模型和梯度、cosine 按 epoch 更新至 0，无学习率 warmup，不保存 checkpoint；Muon 的 Polar Express 正交化内部使用 BF16。CIFAR-10 使用 GPU 0–3，CIFAR-100 使用 GPU 4–7，各自先 Dense 后 GreedyLoRE。

Muon 矩阵 LR **0.02**、momentum 0.95、Nesterov、spectral-norm scaling；scalar / 分类头 AdamW LR 在 CIFAR-10/100 上分别为 **0.005 / 0.0005**，betas=(0.9,0.999)、epsilon=1e-8。Weight decay 为 5e-4，bias 和 BatchNorm 参数为 0。两种数据集沿用原 CIFAR 入口的随机裁剪、水平翻转和 normalization，两方法臂保持一致。

压缩组使用 `top_subspace`、rank 64、EF14、`min_compression_rate=2`、`beta_ef=0`、`error_inherit=0`；CIFAR-10/100 分别在前 500/4000 次 dense 更新后启用，子空间更新间隔分别为 750/1200。首次子空间刷新仍为 dense 聚合，第一次低秩传输是第 502/4002 次更新。卷积梯度展平后压缩，分类头与一维参数保持 dense；压缩发生在 Muon 动量和正交化之前，因此属于近似 Muon。

训练预算、batch、rank、压缩起点和子空间间隔参照 GreedyLoRE Figure 3 / 附录 F；Muon 参数、精度、seed、数据处理及未明确的调度细节是本次适配，结果不是论文 AdamW 数值的直接复现。运行入口为 `pytorch-cifar/train_ddp.py`，配置脚本为 `pytorch-cifar/run_paper_muon.bash`，队列为 `pytorch-cifar/queue_paper_muon.bash`。W&B online 按数据集使用独立 project。

### 结果

| 实验 | final test accuracy (%) | final test loss |
| --- | ---: | ---: |
| CM044 | 94.43 | 0.31748 |
| CM045 | 94.29 | 0.31015 |
| CM046 | 75.91 | 1.27233 |
| CM047 | 74.94 | 1.27259 |

表中指标均为第 **40 epoch / 15,640 步**、完整 **10,000 张 test** 的最终值，不是 best。来源为各 run 的 `all_results.json` 字段 `final_test_accuracy`、`final_test_loss`，与 `metrics.jsonl` 最后一行交叉核对；同时核对 `status=completed`、`epoch`、`update_step`、`test_samples` 和日志 `TRAINING_COMPLETED`。完整 run ID 在 `outputs/` 下按实验编号及方法唯一定位，设置来自各 run 的 `config.json`；两个队列状态均为 `queue_finished`。

### 结论

- CIFAR-10 的 GreedyLoRE 最终 accuracy 比 Dense 低 **0.14 个百分点**，test loss 低 0.00733；CIFAR-100 的 accuracy 低 **0.97 个百分点**，test loss 高 0.00026。
- 本轮 CIFAR-10 的压缩组更接近 Dense，CIFAR-100 的准确率损失更明显。每组只有 seed 1243 一次运行（n=1），未进行 CIFAR 学习率搜索，不能据此声称统计等效或稳定差异。

## GLUE / RoBERTa-base：Table III 相关实验

本地 GLUE Arrow 数据、RoBERTa-base、4 GPU（0、1、2、7）、FP32、每卡 batch 4、10 epoch、seed 1243、cosine 调度和 10% warmup；入口为 `glue/scripts/run_table_iii_muon.bash`。Dense 与 r8/r16 使用相同 Muon 参数：矩阵 LR `2e-4`、scalar LR `5e-5`、momentum 0.95、spectral-norm scaling。r8/r16 使用 `top_subspace`、EF14、第 1,000 次通信迭代开始压缩、投影间隔 200。下表只用完成第 10 个 epoch 的 W&B online `-rerun1`；原 CM009 无 W&B 运行和中断的原 CM010 不参与比较。

数值是**最终 epoch 的 validation 指标**，不是每轮验证中的 best，也不是论文 test。每个单元格后的括号是实验编号。MRPC、QQP 显示 accuracy / F1，STS-B 显示 Pearson / Spearman；其他任务显示单一指标。MNLI 显示 matched / mismatched accuracy。

| 任务及指标 | Dense Muon | r8 + Muon | r16 + Muon |
| --- | ---: | ---: | ---: |
| SST-2，accuracy | 0.91628 (CM009) | 0.93005 (CM010) | 0.93119 (CM011) |
| CoLA，Matthews | 0.61570 (CM012) | 0.61339 (CM013) | 0.61833 (CM014) |
| MRPC，accuracy / F1 | 0.89216 / 0.92308 (CM015) | 0.87990 / 0.91358 (CM016) | 0.89216 / 0.92226 (CM017) |
| STS-B，Pearson / Spearman | 0.90674 / 0.90409 (CM018) | 0.90729 / 0.90444 (CM019) | 0.90645 / 0.90379 (CM020) |
| RTE，accuracy | 0.75090 (CM021) | 0.72924 (CM022) | 0.73285 (CM023) |
| QNLI，accuracy | 0.92422 (CM024) | 0.92385 (CM025) | 0.91909 (CM026) |
| QQP，accuracy / F1 | 0.91019 / 0.88036 (CM027) | 0.91365 / 0.88533 (CM028) | 0.91378 / 0.88538 (CM029) |
| MNLI，matched / mismatched accuracy | 0.85960 / 0.85724 (CM030) | 0.86816 / 0.86252 (CM031) | 0.86663 / 0.86252 (CM032) |

除 MNLI matched 外，表中数值来自相应 `-rerun1/all_results.json` 的 `eval_accuracy`、`eval_f1`、`eval_matthews_correlation`、`eval_pearson` 或 `eval_spearmanr`。MNLI 的 `eval_accuracy` 是最终 **mismatched**；最终 **matched** 来自对应 `train.log` 的 `[RANK 0] epoch 9, eval metric`。已逐项核对 24 个日志都有该行。入口还保存逐指标 `best_*`，但未保存这些最优值各自的 epoch，故本表不使用 best。

同任务对照中，压缩组有高于 Dense 的任务，也有低于 Dense 的任务，例如 SST-2 两个秩均较高，RTE 两个秩均较低。所有任务都只有一个 seed，且结果是 validation；不能据此声称稳定增益或复现论文 test 指标。本次也未统计与 Muon 内部 collective 同口径的总通信量。
