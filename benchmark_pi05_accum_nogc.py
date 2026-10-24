#!/usr/bin/env python3
"""Single-A800 PI05 accumulation benchmark for LeRobot 0.6.1.

Uses the installed trainer, dataset factory, model, processors and optimizer.
Transforms a copy of the training function in memory; installed files are not
edited. Checkpointing, resume, distributed execution and Hub upload are disabled.
This is a throughput/gradient-update benchmark, not a formal training launcher.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

RUN_RECORDS = []


def accumulated_update(metrics, policy, optimizer, grad_clip_norm, accelerator,
                       next_batch, accumulation_steps, expected_micro_batch,
                       lr_scheduler=None):
    """Mean micro losses -> one clip, optimizer step and scheduler step."""
    import torch

    if accelerator.num_processes != 1 or accelerator.gradient_accumulation_steps != 1:
        raise RuntimeError("Requires one process and Accelerate accumulation=1; this script owns accumulation.")
    started = time.perf_counter()
    policy.train()
    optimizer.zero_grad(set_to_none=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    losses = []
    scalar_outputs = {}
    loading_s = 0.0
    for micro in range(accumulation_steps):
        load_started = time.perf_counter()
        batch = next_batch()
        loading_s += time.perf_counter() - load_started
        # All micro batches must have equal size for loss / accumulation_steps.
        if batch["action"].shape[0] != expected_micro_batch:
            raise RuntimeError("Unexpected partial micro batch; exact effective batch is required.")
        with accelerator.autocast():
            loss, outputs = policy.forward(batch)
        loss_value = float(loss.detach().item())
        if not math.isfinite(loss_value):
            raise FloatingPointError(f"Nonfinite loss at micro batch {micro + 1}: {loss_value}")
        accelerator.backward(loss / accumulation_steps)
        losses.append(loss_value)
        for key, value in (outputs or {}).items():
            if hasattr(value, "numel") and value.numel() == 1:
                value = value.detach().item()
            if isinstance(value, (int, float)) and math.isfinite(value):
                scalar_outputs.setdefault(key, []).append(float(value))
        if (micro + 1) % 8 == 0 or micro + 1 == accumulation_steps:
            print(f"  micro {micro + 1}/{accumulation_steps}, loss={loss_value:.6f}", flush=True)
        del batch, loss, outputs

    limit = grad_clip_norm if grad_clip_norm > 0 else float("inf")
    grad_norm = accelerator.clip_grad_norm_(policy.parameters(), limit)
    grad_value = float(grad_norm.item())
    if not math.isfinite(grad_value):
        raise FloatingPointError(f"Nonfinite accumulated gradient norm: {grad_value}")
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    if lr_scheduler is not None:
        lr_scheduler.step()
    unwrapped = accelerator.unwrap_model(policy, keep_fp32_wrapper=True)
    if hasattr(unwrapped, "update") and callable(unwrapped.update):
        unwrapped.update()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    mean_loss = statistics.mean(losses)
    metrics.loss = mean_loss
    metrics.grad_norm = grad_value
    metrics.lr = optimizer.param_groups[0]["lr"]
    metrics.dataloading_s = loading_s
    metrics.update_s = elapsed - loading_s
    peak_gib = None
    if torch.cuda.is_available():
        peak_gib = torch.cuda.max_memory_allocated() / 1024**3
        metrics.gpu_mem_gb = peak_gib
    mean_outputs = {key: statistics.mean(values) for key, values in scalar_outputs.items()}
    if mean_outputs:
        metrics.update_metrics(mean_outputs)
    record = dict(update=len(RUN_RECORDS) + 1, loss=mean_loss,
                  grad_norm=grad_value, elapsed_s=elapsed, loading_s=loading_s,
                  peak_memory_gib=peak_gib,
                  effective_batch=expected_micro_batch * accumulation_steps)
    RUN_RECORDS.append(record)
    print("ACCUM_UPDATE " + json.dumps(record), flush=True)
    return metrics, mean_outputs


def transform_trainer(source: str) -> ast.Module:
    """Fail closed if the expected 0.6.1 training-function structure differs."""
    tree = ast.parse(source)
    fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "train")
    loops = [node for node in fn.body if isinstance(node, ast.For)
             and ast.unparse(node.iter) == "range(step, cfg.steps)"]
    if len(loops) != 1:
        raise RuntimeError("Unsupported trainer loop; no training was started.")
    loop = loops[0]
    update_index = next(i for i, node in enumerate(loop.body)
                        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
                        and ast.unparse(node.value.func) == "update_policy")
    if update_index != 5:
        raise RuntimeError("Unexpected batch preprocessing structure; refusing to patch.")
    originals = loop.body[:update_index]
    expected = ["start_time", "batch", None, "batch", "train_tracker.dataloading_s"]
    for node, target in zip(originals, expected, strict=True):
        if target is not None and (not isinstance(node, ast.Assign)
                                   or ast.unparse(node.targets[0]) != target):
            raise RuntimeError("Unexpected preprocessing statement.")

    # Capture the native data iterator and processor. Keep native uint8 conversion.
    closure = ast.parse("def _next_accumulation_batch():\n    pass\n").body[0]
    closure.body = originals[1:4] + [ast.Return(ast.Name("batch", ast.Load()))]
    fn.body.insert(fn.body.index(loop), closure)
    replacement = ast.parse("""
train_tracker, _ = _pi05_accumulated_update(
    train_tracker, policy, optimizer, cfg.optimizer.grad_clip_norm, accelerator,
    _next_accumulation_batch, _pi05_accumulation_steps, cfg.batch_size,
    lr_scheduler=lr_scheduler,
)
""").body
    loop.body = replacement + loop.body[update_index + 1:]

    changed = {"effective_bs": 0, "effective_batch_size": 0, "tracker": 0, "loader": 0}
    for node in ast.walk(fn):
        if not isinstance(node, ast.Assign):
            continue
        name = ast.unparse(node.targets[0])
        if name in ("effective_bs", "effective_batch_size"):
            node.value = ast.BinOp(node.value, ast.Mult(), ast.Name("_pi05_accumulation_steps", ast.Load()))
            changed[name] += 1
        if name == "train_tracker" and isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == "MetricsTracker":
            node.value.args[0] = ast.parse("cfg.batch_size * _pi05_accumulation_steps", mode="eval").body
            changed["tracker"] += 1
        if name == "dataloader" and isinstance(node.value, ast.Call):
            for kw in node.value.keywords:
                if kw.arg == "drop_last":
                    kw.value = ast.Constant(True)
                    changed["loader"] += 1
    if changed != dict(effective_bs=1, effective_batch_size=1, tracker=1, loader=1):
        raise RuntimeError(f"Unexpected trainer structure: {changed}")

    # Make the earlier native batch-size log agree with the accumulation math.
    for node in ast.walk(fn):
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "logging.info" and node.args:
            if "Effective batch size:" in ast.unparse(node.args[0]):
                node.args[0] = ast.parse(
                    'f"Effective batch size: {cfg.batch_size} x {_pi05_accumulation_steps} x {num_processes} = {effective_bs}"',
                    mode="eval").body
    return ast.fix_missing_locations(tree)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=8, help="Per-micro-batch samples")
    parser.add_argument("--accumulation-steps", type=int, default=32)
    parser.add_argument("--steps", type=int, default=3, help="Optimizer updates, limited to 20")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--base", default="/root/autodl-tmp/pi05/models/pi05_base")
    parser.add_argument("--dataset-root", default="/root/autodl-tmp/pi05/datasets/work/so101_yellow_cube_to_tray_v21")
    parser.add_argument("--report", default="/root/autodl-tmp/pi05/logs/accum_benchmark.json")
    args = parser.parse_args()
    if min(args.batch_size, args.accumulation_steps) < 1 or not 1 <= args.steps <= 20:
        parser.error("Batch/accumulation must be positive; benchmark steps must be 1..20")
    if args.num_workers < 0:
        parser.error("num-workers must be nonnegative")
    if os.environ.get("WORLD_SIZE", "1") != "1":
        parser.error("Only single-GPU execution is supported")
    if os.environ.get("ACCELERATE_GRADIENT_ACCUMULATION_STEPS", "1") != "1":
        parser.error("Unset ACCELERATE_GRADIENT_ACCUMULATION_STEPS; the script manages accumulation")
    version = importlib.metadata.version("lerobot")
    if version != "0.6.1":
        raise RuntimeError(f"Expected LeRobot 0.6.1, found {version}")
    for path in (Path(args.base) / "model.safetensors", Path(args.dataset_root) / "meta/info.json"):
        if not path.is_file():
            raise FileNotFoundError(path)
    meta = json.loads((Path(args.dataset_root) / "meta/info.json").read_text())
    if (meta["total_episodes"], meta["total_frames"], meta["fps"]) != (50, 30969, 30):
        raise RuntimeError("Dataset metadata differs from the audited 50-episode dataset")
    os.environ.setdefault("HF_HOME", "/root/autodl-tmp/pi05/cache/huggingface")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"

    import torch
    from lerobot.scripts import lerobot_train as native
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one CUDA GPU for this benchmark")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 support is required")
    source = inspect.getsource(inspect.unwrap(native.train))
    tree = transform_trainer(source)
    namespace = dict(vars(native))
    namespace.update(_pi05_accumulated_update=accumulated_update,
                     _pi05_accumulation_steps=args.accumulation_steps)
    exec(compile(tree, "<pi05-benchmark-trainer>", "exec", dont_inherit=True), namespace)
    episodes = [i for i in range(50) if i not in (0, 17, 26, 34, 44)]
    run_name = "accum_benchmark_" + time.strftime("%Y%m%d_%H%M%S")
    options = {
        "dataset.repo_id": "local/so101_yellow_cube_to_tray",
        "dataset.root": args.dataset_root, "dataset.episodes": json.dumps(episodes),
        "dataset.video_backend": "torchcodec", "dataset.eval_split": 0.0,
        "policy.type": "pi05", "policy.pretrained_path": args.base,
        "policy.device": "cuda", "policy.dtype": "bfloat16",
        "policy.freeze_vision_encoder": "false", "policy.train_expert_only": "false",
        "policy.gradient_checkpointing": "false", "policy.compile_model": "false",
        "policy.use_relative_actions": "false", "policy.push_to_hub": "false",
        "policy.normalization_mapping": json.dumps(dict(VISUAL="IDENTITY", STATE="QUANTILES", ACTION="QUANTILES")),
        "policy.optimizer_lr": 5e-5,
        "batch_size": args.batch_size, "num_workers": args.num_workers,
        "prefetch_factor": 1, "persistent_workers": "true",
        "steps": args.steps, "log_freq": 1, "env_eval_freq": 0, "eval_steps": 0,
        "save_checkpoint": "false", "wandb.enable": "false", "seed": 42,
        "job_name": run_name, "output_dir": "/root/autodl-tmp/pi05/outputs/" + run_name,
    }
    print(f"Benchmark: micro={args.batch_size}, accumulation={args.accumulation_steps}, "
          f"effective={args.batch_size * args.accumulation_steps}, optimizer updates={args.steps}", flush=True)
    print("Exact-size micro batches; the incomplete final batch of each epoch is dropped. No checkpoints are written.", flush=True)
    sys.argv = [str(Path(native.__file__))] + [f"--{key}={value}" for key, value in options.items()]
    RUN_RECORDS.clear()
    namespace["train"]()
    if len(RUN_RECORDS) != args.steps:
        raise RuntimeError("Optimizer-update count does not match requested benchmark steps")
    stable = RUN_RECORDS[1:] or RUN_RECORDS
    seconds = statistics.mean(record["elapsed_s"] for record in stable)
    effective = args.batch_size * args.accumulation_steps
    report = dict(status="passed", lerobot_version=version, gradient_checkpointing=False,
                  native_trainer_sha256=hashlib.sha256(source.encode()).hexdigest(),
                  micro_batch=args.batch_size, accumulation_steps=args.accumulation_steps,
                  effective_batch=effective, updates=RUN_RECORDS,
                  steady_update_seconds=seconds, steady_samples_per_second=effective / seconds,
                  estimated_30000_updates_hours=seconds * 30000 / 3600,
                  estimate_excludes="startup, validation, checkpoint I/O, long-run variability",
                  normalization="QUANTILES from existing dataset metadata",
                  validation_episodes=[0, 17, 26, 34, 44],
                  incomplete_micro_batch="dropped at each epoch boundary",
                  checkpoint_saved=False)
    dest = Path(args.report)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2) + "\n")
    print("\nBENCHMARK SUMMARY", flush=True)
    print(f"Steady optimizer update: {seconds:.3f} s; throughput: {effective / seconds:.2f} samples/s")
    print(f"30,000 optimizer updates: approximately {seconds * 30000 / 3600:.2f} hours (excluding eval/save)")
    print(f"Report: {dest}")


if __name__ == "__main__":
    main()
