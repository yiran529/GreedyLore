"""Table V wall-clock measurement using the repository's C4/DDP paths."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import time


def summarize_timing(rank_elapsed, rank_steps, expected_steps):
    if expected_steps <= 0 or not rank_elapsed or len(rank_elapsed) != len(rank_steps):
        raise ValueError('Invalid timing window')
    if any(len(steps) != expected_steps for steps in rank_steps):
        raise ValueError('Incomplete timing window')
    if any(not math.isfinite(t) or t <= 0 for t in rank_elapsed):
        raise ValueError('Nonfinite or nonpositive elapsed time')
    if any(not math.isfinite(t) or t <= 0 for steps in rank_steps for t in steps):
        raise ValueError('Nonfinite or nonpositive iteration time')
    slowest = max(range(len(rank_elapsed)), key=rank_elapsed.__getitem__)
    return {
        'measured_steps': expected_steps,
        'mean_iteration_seconds': rank_elapsed[slowest] / expected_steps,
        'rank_elapsed_seconds': rank_elapsed,
        'slowest_rank': slowest,
        'slowest_rank_iteration_median_seconds': statistics.median(rank_steps[slowest]),
        'rank_iteration_seconds': rank_steps,
    }


def build_timing_optimizer(model, args):
    import torch

    if args.optimizer == 'adamw':
        return torch.optim.AdamW(
            model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2),
            eps=args.eps, weight_decay=args.weight_decay,
        )

    from optimizer import build_muon_optimizer
    return build_muon_optimizer(
        model, lr=args.lr, scalar_lr=args.muon_scalar_lr, mu=args.muon_mu,
        weight_decay=args.weight_decay,
        scalar_weight_decay=args.muon_scalar_weight_decay,
        scalar_betas=(args.muon_scalar_beta1, args.muon_scalar_beta2),
        scalar_epsilon=args.muon_scalar_eps, muon_epsilon=args.muon_epsilon,
        adjust_lr=None if args.muon_adjust_lr == 'none' else args.muon_adjust_lr,
        compile_orthogonalization=args.muon_compile,
        distributed_orthogonalization=not args.muon_local_orthogonalization,
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model_config', required=True)
    parser.add_argument('--dataset_path', default='c4/c4_en')
    parser.add_argument('--tokenizer_path', default='t5-base')
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--max_length', type=int, default=256)
    parser.add_argument('--dtype', choices=['bfloat16', 'float32'], default='bfloat16')
    parser.add_argument('--warmup_iterations', type=int, default=1000)
    parser.add_argument('--measured_iterations', type=int, default=500)
    parser.add_argument('--scheduler_steps', type=int, required=True)
    parser.add_argument('--lr_warmup_steps', type=int, required=True)
    parser.add_argument('--activation_checkpointing', action='store_true')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=1243)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--eps', type=float, default=1e-8)
    parser.add_argument('--weight_decay', type=float, default=0.)
    parser.add_argument('--grad_clipping', type=float, default=1.)
    parser.add_argument('--optimizer', choices=['adamw', 'muon'], default='muon')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--dry-run', action='store_true')
    from comm_hooks.utils import add_comm_hook_args
    from optimizer import add_muon_args
    add_comm_hook_args(parser)
    add_muon_args(parser, scalar_lr_default=0.001, scalar_weight_decay_default=0.)
    args = parser.parse_args()
    if args.measured_iterations <= 0 or args.warmup_iterations < 0:
        parser.error('Invalid timing budget')
    if args.compressor not in ['none', 'powersgd', 'top_subspace']:
        parser.error('Use none, powersgd, or top_subspace')
    if args.compressor != 'none' and args.start_compress_iter > args.warmup_iterations:
        parser.error('Compression must be enabled throughout the measurement window')
    return args


def main():
    args = parse_args()
    if args.dry_run:
        print(json.dumps(vars(args), indent=2))
        return

    import numpy as np
    import torch
    import torch.distributed as dist
    import datasets.distributed
    from transformers import AutoConfig, AutoTokenizer
    from c4.pept_utils.c4_data import load_c4_split
    from c4.pept_utils.dataloader import PreprocessedIterableDataset
    from c4.pept_utils.modeling_llama import LlamaForCausalLM
    from c4.pept_utils.training_utils import get_scheculer
    from comm_hooks.utils import register_comm_hook_for_ddp_model

    rank, local_rank, world_size = (int(os.environ[key]) for key in ['RANK', 'LOCAL_RANK', 'WORLD_SIZE'])
    torch.cuda.set_device(local_rank)
    dist.init_process_group('nccl')
    device = torch.device('cuda', local_rank)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    output = Path(args.output_dir)
    if rank == 0:
        output.mkdir(parents=True, exist_ok=False)
        config = vars(args).copy()
        config.update(world_size=world_size, total_batch_size=args.batch_size * world_size,
                      gradient_accumulation=1, status='running',
                      measurement=f'max_rank(continuous_{args.measured_iterations}_iteration_wall_time)/{args.measured_iterations}; includes data fetch, H2D, forward/backward, DDP hook, clipping, optimizer, scheduler and zero_grad',
                      torch_version=torch.__version__, gpu=torch.cuda.get_device_name(),
                      git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                      nccl_environment={key: os.environ.get(key) for key in ['NCCL_P2P_DISABLE', 'NCCL_SHM_DISABLE', 'NCCL_DEBUG', 'NCCL_CUMEM_HOST_ENABLE']})
        paths = [Path(__file__), Path(args.model_config), Path('optimizer/muon.py'), Path('optimizer/muon_utils.py'), Path('comm_hooks/utils.py'), Path('comm_hooks/subspace_hook.py'), Path('comm_hooks/powerSGD_hook.py'), Path('c4/pept_utils/modeling_llama.py'), Path('c4/pept_utils/dataloader.py'), Path('c4/pept_utils/c4_data.py')]
        config['code_sha256'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        (output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    dist.barrier()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, model_max_length=args.max_length, local_files_only=True)
    data = load_c4_split(args.dataset_path, 'train').shuffle(seed=42)
    data = datasets.distributed.split_dataset_by_node(data, rank=rank, world_size=world_size)
    dataset = PreprocessedIterableDataset(data, tokenizer, args.batch_size, args.max_length)
    loader = torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=args.workers)
    batches = iter(loader)
    model_config = AutoConfig.from_pretrained(args.model_config)
    model_config.pad_token_id = tokenizer.pad_token_id
    model_config.use_cache = False
    model = LlamaForCausalLM(model_config)
    if args.activation_checkpointing:
        model.gradient_checkpointing_enable()
    model.to(device=device, dtype=getattr(torch, args.dtype))
    model.train()
    parameter_count = sum(p.numel() for p in model.parameters())
    optimizer = build_timing_optimizer(model, args)
    scheduler = get_scheculer(optimizer=optimizer, scheduler_type='cosine',
                              num_training_steps=args.scheduler_steps,
                              warmup_steps=args.lr_warmup_steps, min_lr_ratio=0.1)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)
    register_comm_hook_for_ddp_model(model, dist.group.WORLD, args)
    parameters = [p for p in model.parameters() if p.requires_grad]
    if rank == 0:
        config.update(parameter_count=parameter_count, model=model_config.to_dict(),
                      optimizer_groups=[{k: v for k, v in g.items() if k != 'params'} | {'parameter_count': sum(p.numel() for p in g['params'])} for g in optimizer.param_groups])
        (output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
        print(f'INITIALIZED params={parameter_count} rank={rank} compressor={args.compressor}', flush=True)
    step_seconds, losses = [], []
    window_start = None
    for step in range(args.warmup_iterations + args.measured_iterations):
        if step == args.warmup_iterations:
            dist.barrier()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            if rank == 0:
                print(f'MEASUREMENT_START step={step + 1}', flush=True)
            window_start = time.perf_counter()
        iteration_start = time.perf_counter()
        batch = next(batches)  # Data exhaustion is an error; no sample recycling.
        if batch['input_ids'].shape[0] != args.batch_size:
            raise RuntimeError('Partial batch during timing')
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch['input_ids'].clone()
        labels[labels == tokenizer.pad_token_id] = -100
        loss = model(**batch, labels=labels).loss
        loss.backward()
        if args.grad_clipping:
            torch.nn.utils.clip_grad_norm_(parameters, args.grad_clipping)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        torch.cuda.synchronize()
        iteration_end = time.perf_counter()
        if step >= args.warmup_iterations:
            step_seconds.append(iteration_end - iteration_start)
            losses.append(loss.detach())
        elif step == 0 or (step + 1) % 100 == 0:
            if rank == 0:
                print(f'WARMUP step={step+1} loss={loss.item():.6f}', flush=True)
    elapsed = iteration_end - window_start
    local_losses = torch.stack(losses).float().cpu().tolist()
    if not all(math.isfinite(value) for value in local_losses):
        raise RuntimeError('Nonfinite loss during measurement')
    local_result = dict(elapsed=elapsed, steps=step_seconds, losses=local_losses,
                        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                        peak_reserved_bytes=torch.cuda.max_memory_reserved())
    gathered = [None] * world_size
    dist.all_gather_object(gathered, local_result)
    if rank == 0:
        result = summarize_timing([r['elapsed'] for r in gathered], [r['steps'] for r in gathered], args.measured_iterations)
        result.update(status='completed', optimizer=args.optimizer, compressor=args.compressor,
                      parameter_count=parameter_count, warmup_iterations=args.warmup_iterations,
                      first_measured_update=args.warmup_iterations + 1,
                      last_measured_update=args.warmup_iterations + args.measured_iterations,
                      rank_losses=[r['losses'] for r in gathered],
                      rank_peak_allocated_bytes=[r['peak_allocated_bytes'] for r in gathered],
                      rank_peak_reserved_bytes=[r['peak_reserved_bytes'] for r in gathered])
        (output / 'all_results.json').write_text(json.dumps(result, indent=2) + '\n')
        print(f'TIMING_COMPLETED mean_iteration_seconds={result["mean_iteration_seconds"]:.6f}', flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
