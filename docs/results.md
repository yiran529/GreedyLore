# GreedyLoRE + Muon 实验结果（2026-09-29 核对）

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
