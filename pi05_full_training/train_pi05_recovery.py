#!/usr/bin/env python3
"""Single-GPU PI05 full fine-tuning for the audited LeRobot 0.6.1 trainer.
Micro batch 8, accumulation 32, BF16, no gradient checkpointing.
Uses the installed native model, processors, optimizer and checkpoint routines.
The trainer adaptation is in memory; site-packages are never edited.
"""
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
import shutil
import tempfile
from dataclasses import dataclass

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
    used_lr = optimizer.param_groups[0]["lr"]
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
    record["lr_used"] = used_lr
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
    closure.body = originals[1:3] + ast.parse('batch["task"] = [\'Pick up the yellow cube from inside the black outlined area and place it in the gray-blue tray.\'] * batch["action"].shape[0]').body + originals[3:4] + [ast.Return(ast.Name("batch", ast.Load()))]
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


NATIVE_SHA256 = "f2c2f33358951cede0ebc96604209c515d0daff26821867354feec81725bb4de"
MICRO = 8
ACCUM = 32
TOTAL = 2000
WARMUP = 100
DECAY_START = 100
MIN_LR_RATIO = 0.1
LR = 1e-5
TRAIN_FRAMES = 45843
VALIDATION_EPISODES = [0, 17, 26, 34, 44]
TRAIN_EPISODES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 29, 30, 31, 32, 33, 35, 36, 37, 38, 39, 40, 41, 42, 43, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87]


def wsd_factor(completed_updates, warmup=WARMUP, decay_start=DECAY_START,
               total=TOTAL, floor=MIN_LR_RATIO):
    """Linear warmup, then cosine; argument names retained for config compatibility."""
    u = completed_updates + 1
    if u <= warmup:
        return u / warmup
    progress = min(1.0, (u - warmup) / (total - warmup))
    return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))


def accumulated_sampler_state(step, sampler_length, batch_size, num_processes):
    """Native sampler cursor with accumulation AND dropped partial batches."""
    if num_processes != 1 or batch_size != MICRO:
        raise RuntimeError("Resume requires the original single GPU and micro batch 8")
    batches_per_epoch = sampler_length // batch_size
    if batches_per_epoch < 1 or step < 0:
        raise ValueError("Invalid sampler size or checkpoint step")
    epoch, batch_offset = divmod(step * ACCUM, batches_per_epoch)
    return {"epoch": epoch, "start_index": batch_offset * batch_size}


def transform_formal(source):
    tree = transform_trainer(source)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "train")
    loop = next(n for n in fn.body if isinstance(n, ast.For)
                and ast.unparse(n.iter) == "range(step, cfg.steps)")
    loop.iter = ast.parse("range(step, min(cfg.steps, _formal_stop_step))", mode="eval").body
    # Keep cfg.steps=30000 for scheduler and checkpoint numbering. Only this
    # invocation stops early. Native saving remains after the completed update.
    counts = {"setup": 0, "record": 0, "save": 0, "generator": 0, "sampler": 0}
    for i, n in enumerate(fn.body):
        if isinstance(n, ast.Expr) and ast.unparse(n) == "cfg.validate()":
            fn.body.insert(i + 1, ast.parse("_formal_setup(cfg)").body[0])
            counts["setup"] += 1
            break
    for i, n in enumerate(loop.body):
        if isinstance(n, ast.AugAssign) and ast.unparse(n.target) == "step":
            loop.body.insert(i + 1, ast.parse("_formal_record(step)").body[0])
            counts["record"] += 1
            break
    for n in ast.walk(fn):
        if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "is_saving_step":
            n.value = ast.BoolOp(ast.Or(), [n.value, ast.parse("step == _formal_stop_step", mode="eval").body])
            counts["save"] += 1
        if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "dataloader":
            # Isolate iterator base-seed draws from the global training RNG.
            n.value.keywords.append(ast.keyword(arg="generator", value=ast.parse(
                "torch.Generator().manual_seed(cfg.seed if cfg.seed is not None else 0)", mode="eval").body))
            counts["generator"] += 1
        if isinstance(n, ast.Call) and ast.unparse(n.func) == "compute_sampler_state":
            counts["sampler"] += 1
        if isinstance(n, ast.Call) and ast.unparse(n.func) == "tqdm":
            for kw in n.keywords:
                if kw.arg == "total":
                    kw.value = ast.parse("min(cfg.steps, _formal_stop_step) - step", mode="eval").body
    if counts != dict(setup=1, record=1, save=1, generator=1, sampler=1):
        raise RuntimeError(f"Unsupported trainer structure: {counts}")
    return ast.fix_missing_locations(tree)


def tree_bytes(path):
    return sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file() and not p.is_symlink())


def complete_checkpoints(root):
    result = []
    if not root.is_dir():
        return result
    for p in root.iterdir():
        marker = p / "formal_state.json"
        if p.is_dir() and not p.is_symlink() and p.name.isdigit() and marker.is_file():
            state = json.loads(marker.read_text())
            if state.get("native_trainer_sha256") == NATIVE_SHA256:
                result.append((state["step"], p))
    return sorted(result)


def validate_checkpoint(path):
    path = Path(path)
    pm = path / "pretrained_model"
    for name in ["train_config.json", "config.json"]:
        if not (pm / name).is_file():
            raise RuntimeError(f"Checkpoint missing {pm / name}")
    if not list(pm.glob("*.safetensors")):
        raise RuntimeError("Checkpoint has no model safetensors")
    for pattern in ["*preprocessor*.json", "*postprocessor*.json"]:
        if not list(pm.glob(pattern)):
            raise RuntimeError(f"Checkpoint missing processor config: {pattern}")
    state = path / "training_state"
    if not state.is_dir() or not any(p.is_file() for p in state.rglob("*")):
        raise RuntimeError("Checkpoint missing native training_state")


def atomic_checkpoint(native_save, kwargs, recipe, script_path):
    target = Path(kwargs["checkpoint_dir"])
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint: {target}")
    # Actual tensor sizes give a better disk bound than GPU memory usage.
    policy, optimizer = kwargs["policy"], kwargs["optimizer"]
    model_bytes = sum(t.numel() * t.element_size() for t in policy.state_dict().values())
    def tensor_bytes(obj):
        if hasattr(obj, "numel") and hasattr(obj, "element_size"):
            return obj.numel() * obj.element_size()
        if isinstance(obj, dict):
            return sum(tensor_bytes(v) for v in obj.values())
        if isinstance(obj, (tuple, list)):
            return sum(tensor_bytes(v) for v in obj)
        return 0
    needed = int((model_bytes + tensor_bytes(optimizer.state_dict())) * 1.15) + 1024**3
    free = shutil.disk_usage(target.parent).free
    if free < needed:
        raise RuntimeError(f"Checkpoint needs about {needed / 1024**3:.1f} GiB free; only {free / 1024**3:.1f} GiB available")
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.writing-", dir=target.parent))
    try:
        native_save(**{**kwargs, "checkpoint_dir": staging})
        validate_checkpoint(staging)
        (staging / "formal_recipe.json").write_text(json.dumps(recipe, indent=2) + "\n")
        state = dict(step=kwargs["step"], micro_batch=MICRO, accumulation_steps=ACCUM,
                     effective_batch=MICRO * ACCUM, native_trainer_sha256=NATIVE_SHA256,
                     sampler_state=accumulated_sampler_state(kwargs["step"], TRAIN_FRAMES, MICRO, 1))
        (staging / "formal_state.json").write_text(json.dumps(state, indent=2) + "\n")
        shutil.copy2(script_path, staging / "train_pi05_full.py")
        staging.rename(target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def prune_checkpoints(root, keep):
    if keep < 1:
        raise ValueError("At least one checkpoint must be retained")
    items = complete_checkpoints(root)
    latest = (root / "last").resolve()
    for _, p in items[:-keep]:
        if p.resolve() == latest:
            raise RuntimeError("Refusing to prune the last checkpoint")
        shutil.rmtree(p)
        print(f"Removed older complete checkpoint: {p}", flush=True)


def register_wsd():
    from lerobot.optim.schedulers import LRSchedulerConfig
    from torch.optim.lr_scheduler import LambdaLR

    @LRSchedulerConfig.register_subclass("pi05_wsd")
    @dataclass
    class PI05WSDConfig(LRSchedulerConfig):
        num_warmup_steps: int = WARMUP
        decay_start_step: int = DECAY_START
        total_steps: int = TOTAL
        min_lr_ratio: float = MIN_LR_RATIO

        def build(self, optimizer, num_training_steps):
            if num_training_steps != self.total_steps:
                raise ValueError("Keep the full WSD schedule; use --until to pause")
            def factor(k):
                return wsd_factor(k, self.num_warmup_steps, self.decay_start_step,
                                  self.total_steps, self.min_lr_ratio)
            return LambdaLR(optimizer, factor)
    return PI05WSDConfig


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--until", type=int, default=2000, help="Absolute optimizer step to stop AFTER saving")
    ap.add_argument("--resume", action="store_true", help="Resume the last complete checkpoint")
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--keep-checkpoints", type=int, default=4)
    ap.add_argument("--output-dir", default='/root/autodl-tmp/pi05/outputs/pi05_recovery_cosine_v1')
    ap.add_argument("--base", default='/root/autodl-tmp/pi05/outputs/pi05_full_wsd_v1/checkpoints/002000/pretrained_model')
    ap.add_argument("--dataset-root", default='/root/autodl-tmp/pi05/datasets/work/so101_pick_place_recovery_88_v30')
    ap.add_argument("--check-only", action="store_true", help="Check source, dataset, config parsing and video decoder; do not load model")
    args = ap.parse_args()
    if not 1 <= args.until <= TOTAL or min(args.save_every, args.keep_checkpoints) < 1:
        ap.error("Require 1 <= until <= 30000, positive save-every and keep-checkpoints")
    if os.environ.get("WORLD_SIZE", "1") != "1" or os.environ.get("ACCELERATE_GRADIENT_ACCUMULATION_STEPS", "1") != "1":
        ap.error("This launcher owns accumulation and supports exactly one GPU/process")
    os.environ.setdefault("HF_HOME", "/root/autodl-tmp/pi05/cache/huggingface")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    if importlib.metadata.version("lerobot") != "0.6.1":
        raise RuntimeError("This launcher requires the audited LeRobot 0.6.1")

    import torch
    from lerobot.scripts import lerobot_train as native
    import av
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Expose exactly one BF16-capable CUDA GPU")
    source = inspect.getsource(inspect.unwrap(native.train))
    digest = hashlib.sha256(source.encode()).hexdigest()
    if digest != NATIVE_SHA256:
        raise RuntimeError(f"Native trainer changed: {digest}; no training started")
    tree = transform_formal(source)
    register_wsd()

    dataset_root, base = Path(args.dataset_root).resolve(), Path(args.base).resolve()
    meta_path, stats_path = dataset_root / "meta/info.json", dataset_root / "meta/stats.json"
    for path in [meta_path, stats_path, base / "model.safetensors"]:
        if not path.is_file():
            raise FileNotFoundError(path)
    meta = json.loads(meta_path.read_text())
    if (meta["total_episodes"], meta["total_frames"], meta["fps"]) != (88, 48729, 30):
        raise RuntimeError("Expected 88 episodes, 48729 frames, 30 FPS")
    stats = json.loads(stats_path.read_text())
    # 本轮归一化参数由 checkpoint 的 processors 加载。
    # 合并数据的统计量不用于覆盖它们。
    for pattern in ("*preprocessor*.json", "*postprocessor*.json"):
        if not list(base.glob(pattern)):
            raise RuntimeError(f"Checkpoint missing processor config: {pattern}")
    print("Normalization source: checkpoint processors; merged quantiles are not required", flush=True)
    video = next(dataset_root.rglob("*.mp4"), None)
    if video is None:
        raise FileNotFoundError("No MP4 video found")
    with av.open(str(video)) as container:
        frame = next(container.decode(video=0))
        print(f"PyAV decoder OK: {frame.width}x{frame.height}", flush=True)
    del frame

    output = Path(args.output_dir).resolve()
    recipe = dict(format_version=1, native_trainer_sha256=digest,
                  launcher_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  dataset_root=str(dataset_root), base=str(base),
                  info_sha256=hashlib.sha256(meta_path.read_bytes()).hexdigest(),
                  stats_sha256=hashlib.sha256(stats_path.read_bytes()).hexdigest(),
                  train_episodes=TRAIN_EPISODES, validation_episodes=VALIDATION_EPISODES,
                  micro_batch=MICRO, accumulation_steps=ACCUM, effective_batch=MICRO * ACCUM,
                  dtype="bfloat16", gradient_checkpointing=False, lr=LR,
                  wsd=dict(warmup=WARMUP, decay_start=DECAY_START, total=TOTAL, min_lr_ratio=MIN_LR_RATIO),
                  seed=42, num_workers=0, drop_last=True)
    recipe["schedule_name"] = "linear_warmup_cosine"
    recipe["task_override"] = 'Pick up the yellow cube from inside the black outlined area and place it in the gray-blue tray.'
    recipe["normalization_source"] = str(base)
    recipe["video_backend"] = "pyav"
    start_step = 0
    if args.resume:
        checkpoint = (output / "checkpoints/last").resolve()
        validate_checkpoint(checkpoint)
        if json.loads((checkpoint / "formal_recipe.json").read_text()) != recipe:
            raise RuntimeError("Checkpoint recipe/source/data differ; refusing an inconsistent resume")
        saved = json.loads((checkpoint / "formal_state.json").read_text())
        start_step = saved["step"]
        if args.until <= start_step:
            raise ValueError(f"Already at step {start_step}; --until must be larger")
        config_path = checkpoint / "pretrained_model/train_config.json"
        options = {"config_path": str(config_path), "resume": "true", "output_dir": str(output),
                   "save_freq": args.save_every}
        # Account for the next complete save before old checkpoints are pruned.
        count = len(complete_checkpoints(output / "checkpoints"))
        n_saves = args.until // args.save_every - start_step // args.save_every
        n_saves += int(args.until % args.save_every != 0)
        copies = min(n_saves, max(1, args.keep_checkpoints - count + 1))
        required = int(tree_bytes(checkpoint) * copies * 1.1) + 1024**3
        free = shutil.disk_usage(output).free
        if free < required:
            raise RuntimeError(f"Upcoming checkpoint saves need about {required / 1024**3:.1f} GiB free; available {free / 1024**3:.1f} GiB")
    else:
        if output.exists():
            raise FileExistsError(f"Use --resume for a complete run, or a new --output-dir: {output}")
        options = {
            "dataset.repo_id": "local/so101_pick_place_recovery_88_v30", "dataset.root": str(dataset_root),
            "dataset.episodes": json.dumps(TRAIN_EPISODES), "dataset.video_backend": "pyav",
            "dataset.eval_split": 0.0, "dataset.image_transforms.enable": "false",
            "policy.type": "pi05", "policy.pretrained_path": str(base),
            "policy.device": "cuda", "policy.dtype": "bfloat16",
            "policy.freeze_vision_encoder": "false", "policy.train_expert_only": "false",
            "policy.gradient_checkpointing": "false", "policy.compile_model": "false",
            "policy.use_relative_actions": "false", "policy.push_to_hub": "false",
            "policy.normalization_mapping": json.dumps(dict(VISUAL="IDENTITY", STATE="QUANTILES", ACTION="QUANTILES")),
            "policy.optimizer_lr": LR, "use_policy_training_preset": "false",
            "optimizer.type": "adamw", "optimizer.lr": LR, "optimizer.betas": "[0.9,0.95]",
            "optimizer.eps": 1e-8, "optimizer.weight_decay": 0.01, "optimizer.grad_clip_norm": 1.0,
            "scheduler.type": "pi05_wsd", "scheduler.num_warmup_steps": WARMUP,
            "scheduler.decay_start_step": DECAY_START, "scheduler.total_steps": TOTAL,
            "scheduler.min_lr_ratio": MIN_LR_RATIO,
            "batch_size": MICRO, "num_workers": 0, "prefetch_factor": 1,
            "persistent_workers": "false", "steps": TOTAL, "log_freq": 10,
            "env_eval_freq": 0, "eval_steps": 0, "save_checkpoint": "true",
            "save_freq": args.save_every, "save_checkpoint_to_hub": "false",
            "wandb.enable": "false", "seed": 42, "job_name": "pi05_recovery_cosine_v1", "output_dir": str(output),
        }
    if not args.resume:
        free = shutil.disk_usage(output.parent).free
        required = int(tree_bytes(base.parent) * args.keep_checkpoints * 1.15) + 2 * 1024**3
        print(f"Checkpoint disk estimate: need {required / 1024**3:.1f} GiB; free {free / 1024**3:.1f} GiB", flush=True)
        if free < required:
            raise RuntimeError("Insufficient disk space for four complete checkpoints")
    print(f"FULL FT | micro={MICRO}, accum={ACCUM}, EBS={MICRO * ACCUM}, GC=off", flush=True)
    print(f"COSINE: warmup={WARMUP}, total={TOTAL}, peak={LR}, final={LR * MIN_LR_RATIO}", flush=True)
    print(f"This invocation: {start_step} -> {args.until}; keep last {args.keep_checkpoints} complete checkpoints", flush=True)

    def setup(cfg):
        if cfg.steps != TOTAL or cfg.batch_size != MICRO or cfg.num_workers != 0:
            raise RuntimeError("Unexpected steps, micro batch or loader settings")
        if cfg.dataset.episodes != TRAIN_EPISODES or cfg.dataset.eval_split != 0.0:
            raise RuntimeError("The fixed train/validation episode split changed")
        if cfg.policy.freeze_vision_encoder or cfg.policy.train_expert_only or cfg.policy.gradient_checkpointing:
            raise RuntimeError("Expected full fine-tuning without checkpointing")
        if cfg.policy.dtype != "bfloat16" or cfg.peft is not None or cfg.policy.use_peft:
            raise RuntimeError("Expected BF16 full fine-tuning without PEFT")
        if cfg.scheduler.type != "pi05_wsd" or cfg.optimizer.lr != LR:
            raise RuntimeError("Unexpected LR schedule")
        output.mkdir(parents=True, exist_ok=True)
        (output / "formal_recipe.json").write_text(json.dumps(recipe, indent=2) + "\n")

    def record(step):
        item = {**RUN_RECORDS[-1], "update": step}
        with (output / "updates.jsonl").open("a") as f:
            f.write(json.dumps(item) + "\n")
        if step <= start_step + 2 or step % 10 == 0 or step == args.until:
            print("FULL_UPDATE " + json.dumps(item), flush=True)

    def save(**kwargs):
        atomic_checkpoint(native.save_checkpoint, kwargs, recipe, Path(__file__))

    def update_last(path):
        native.update_last_checkpoint(path)
        prune_checkpoints(output / "checkpoints", args.keep_checkpoints)
        print(f"COMPLETE_CHECKPOINT {path}", flush=True)

    def checkpoint_processors(*args_, **kwargs_):
        if Path(kwargs_["pretrained_path"]).resolve() != base:
            raise RuntimeError("Unexpected processor checkpoint")
        kwargs_.pop("dataset_stats", None)
        for group in ("preprocessor_overrides", "postprocessor_overrides"):
            overrides = kwargs_.get(group, {})
            for name in ("normalizer_processor", "unnormalizer_processor"):
                if name in overrides:
                    overrides[name] = dict(overrides[name])
                    overrides[name].pop("stats", None)
        print("Loading normalization from checkpoint:", base, flush=True)
        return native.make_pre_post_processors(*args_, **kwargs_)

    namespace = dict(vars(native))
    namespace["make_pre_post_processors"] = checkpoint_processors
    namespace.update(_pi05_accumulated_update=accumulated_update, _pi05_accumulation_steps=ACCUM,
                     _formal_stop_step=args.until, _formal_setup=setup, _formal_record=record,
                     compute_sampler_state=accumulated_sampler_state,
                     save_checkpoint=save, update_last_checkpoint=update_last)
    exec(compile(tree, "<pi05-full-trainer>", "exec", dont_inherit=True), namespace)
    sys.argv = [str(Path(native.__file__))] + [f"--{key}={value}" for key, value in options.items()]
    if args.check_only:
        # Exercise the exact CLI parser via the same native wrapper, but return
        # before cfg.validate() creates an output directory or loads a policy.
        @native.parser.wrap()
        def config_check(cfg: native.TrainPipelineConfig):
            assert cfg.scheduler.type == "pi05_wsd"
            assert cfg.batch_size == MICRO and cfg.steps == TOTAL
            print("PREFLIGHT PASSED: source, dataset, PyAV, and training config parser")
        config_check()
        return
    RUN_RECORDS.clear()
    namespace["train"]()
    if len(RUN_RECORDS) != args.until - start_step:
        raise RuntimeError("Completed optimizer-update count differs from requested count")
    checkpoint = (output / "checkpoints/last").resolve()
    validate_checkpoint(checkpoint)
    if json.loads((checkpoint / "formal_state.json").read_text())["step"] != args.until:
        raise RuntimeError("Final checkpoint step mismatch")
    print(f"FINISHED AND SAVED at optimizer step {args.until}: {checkpoint}", flush=True)


if __name__ == "__main__":
    main()

