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
