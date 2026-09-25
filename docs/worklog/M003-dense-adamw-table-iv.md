# M003：Table IV 60M Dense AdamW

## 2026-09-25：追加对照并排队

- 目的：追加论文 Table IV 的 Dense AdamW 对照，和 CM001–CM003 使用相同 C4、推定的 LLaMA 60M 架构、BF16、全局 batch 512、每卡 batch 128、GA=1、seed 1243、10,000 更新步、评估口径及无 checkpoint 设置。
- 实验编号：`CM004-m003-dense-adamw-llama60m-c4-adam_dense-lr2p5e3-bf16-s1243`。
- 超参数：`--optimizer adamw --compressor none --lr 0.0025 --beta1 0.9 --beta2 0.999 --eps 1e-8 --weight_decay 0`；warmup 1,000 步，cosine 降至峰值 LR 的 10%，梯度裁剪 1.0。用户从论文附录给出的 AdamW LR 候选集合 `{0.0015, 0.0025, 0.005}` 中选定单次运行值 0.0025；论文没有指出 Table IV 的 33.44 对应哪个候选值，不能将本次运行称为论文最佳学习率复现。
- 代码：`c4/scripts/run_table_iv_60m_muon.bash adam_dense`；排队入口 `c4/scripts/run_table_iv_60m_adam_after_muon.bash`。W&B online，run name 与实验编号一致。
- 依赖：等 `greedylore_tableiv_60m_bf16` 结束后，检查 CM003 `all_results.json` 中 `update_step=10000`，才启动 CM004；若前三组未全部完成，排队任务退出，不占用 GPU。
- 已创建 `greedylore_tableiv_60m_adam` tmux 排队会话；状态日志为 `outputs/CM004-m003-dense-adamw-llama60m-c4-adam_dense-lr2p5e3-bf16-s1243/queue.log`。当前无结果；完成后按最低 validation PPL 与 Muon 三组比较。

## 2026-09-25：前序任务失败与预算调整

- 前序 CM001 因 C4 数据耗尽及 W&B URL 接口报错而退出，CM002/CM003 未启动；AdamW 排队任务检查不到 CM003 结果，按设计退出，CM004 未启动。
- 用户将预算明确为约 1.1B token 位置（含 padding）。四组脚本及排队完成校验统一改为 8,393 更新步；AdamW LR 仍为 0.0025。数据补足方式待定。

## 2026-09-25：数据量纠正

- 前次 C4 提前耗尽的根因是训练包装器重复执行 worker 切分，现已修复；现有 30 个分片足够四组各自的 8,393 步预算。AdamW 继续排在三组 Muon 后运行。
