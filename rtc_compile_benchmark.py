"""Run on the A800 with RTC server stopped. Does not control a robot."""
import json
import statistics
import time
from pathlib import Path

import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.rtc.configuration_rtc import RTCConfig

CHECKPOINT = '/root/autodl-tmp/pi05/outputs/pi05_recovery_cosine_tc_v1/checkpoints/002000/pretrained_model'
TASK = 'Pick up the yellow cube from inside the black outlined area and place it in the gray-blue tray.'

def main():
    cfg = PI05Config.from_pretrained(CHECKPOINT)
    cfg.pretrained_path = CHECKPOINT
    cfg.device = 'cuda'
    cfg.compile_model = False
    cfg.gradient_checkpointing = False
    cfg.num_inference_steps = 10
    cfg.rtc_config = RTCConfig(enabled=True, execution_horizon=30, max_guidance_weight=10.0)
    assert not cfg.use_relative_actions
    dc = json.loads((Path(CHECKPOINT)/'train_config.json').read_text())['dataset']
    ds = LeRobotDataset(repo_id=dc['repo_id'],root=dc['root'],episodes=[dc['episodes'][0]],video_backend='torchcodec')
    samples = [ds[round((len(ds)-1)*fraction)] for fraction in (.15,.5,.85)]
    observations = []
    for sample in samples:
        observations.append({**{key:sample[key] for key in cfg.input_features},'task':TASK})
    print('Loading checkpoint',CHECKPOINT,flush=True)
    policy = PI05Policy.from_pretrained(CHECKPOINT,config=cfg,strict=True,local_files_only=True).eval()
    policy.init_rtc_processor()
    assert policy._rtc_enabled()
    pre, post = make_pre_post_processors(policy_cfg=cfg,pretrained_path=CHECKPOINT)
    generator = torch.Generator(device='cuda').manual_seed(20261023)
    noises = [torch.randn(1,50,32,generator=generator,device='cuda') for _ in observations]

    @torch.no_grad()
    def run(obs,noise,prefix=None,delay=0):
        torch.cuda.synchronize()
        begin=time.perf_counter()
        batch=pre({k:v.clone() if isinstance(v,torch.Tensor) else v for k,v in obs.items()})
        norm=policy.predict_action_chunk(batch,noise=noise.clone(),prev_chunk_left_over=prefix,
                                        inference_delay=delay,execution_horizon=30)
        saved=norm.detach().clone()
        physical=post(norm.clone()).detach().to('cpu',copy=True)
        saved=saved.to('cpu',copy=True)
        torch.cuda.synchronize()
        ms=(time.perf_counter()-begin)*1000
        assert saved.shape==physical.shape==(1,50,6)
        assert torch.isfinite(saved).all() and torch.isfinite(physical).all()
        return saved,physical,ms

    prefixes=[]
    for obs,noise in zip(observations,noises):
        norm,_,_=run(obs,noise)
        prefixes.append(norm[:,20:].to('cuda').clone())
    delays=[12,18,24]
    cases=[(i,d) for i in range(3) for d in delays]
    # All cases use exactly the same saved inputs on both paths.
    references={}
    for i,d in cases:
        references[i,d]=run(observations[i],noises[i],prefixes[i],d)
    print("\n=== EAGER SELF PARITY ===", flush=True)

    for i, d in cases:
        rn, ra, _ = references[i, d]

        n, a, _ = run(
            observations[i],
            noises[i],
            prefixes[i],
            d
        )
    
        norm_err = (n - rn).abs()
        action_err = (a - ra).abs()
    
        print(json.dumps({
            "frame": i,
            "delay": d,
            "normalized_mae": norm_err.mean().item(),
            "normalized_max": norm_err.max().item(),
            "action_mae": action_err.mean().item(),
            "action_max": action_err.max().item(),
            "per_joint_max": action_err.amax(dim=(0, 1)).tolist()
        }), flush=True)
    eager_times=[]
    for i,d in cases:
        eager_times.append(run(observations[i],noises[i],prefixes[i],d)[2])
    eager_fn=policy.model.denoise_step
    print('Compiling denoise_step only, mode=default; RTC guidance remains eager.',flush=True)
    policy.model.denoise_step=torch.compile(eager_fn,mode='default',fullgraph=False)
    # No reduce-overhead CUDA graphs; do not wrap RTC in inference_mode.
    for i in range(3):
        _,_,ms=run(observations[0],noises[0],prefixes[0],18)
        print(f'COMPILE_WARMUP {i+1}: {ms:.1f} ms',flush=True)
    errors = []

    for i, d in cases:
        n, a, ms = run(
            observations[i],
            noises[i],
            prefixes[i],
            d
        )

        rn, ra, _ = references[i, d]

        # Eager 重复运行，用于测量基准自身波动
        en, ea, _ = run(
            observations[i],
            noises[i],
            prefixes[i],
            d
        ) if False else (None, None, None)

        norm_error = (n - rn).abs()
        action_error = (a - ra).abs()

        norm_ok = torch.allclose(
            n, rn,
            atol=1e-3,
            rtol=1e-3
        )

        action_max = action_error.max().item()
        action_ok = action_max <= 0.1

        joint_mae = action_error.mean(dim=(0, 1)).tolist()
        joint_max = action_error.amax(dim=(0, 1)).tolist()

        row = {
            "frame": i,
            "delay": d,

            "norm_ok": bool(norm_ok),
            "action_ok": bool(action_ok),

            "normalized_mae": norm_error.mean().item(),
            "normalized_max": norm_error.max().item(),

            "action_mae": action_error.mean().item(),
            "action_max": action_max,

            "per_joint_mae": joint_mae,
            "per_joint_max": joint_max,

            "passed": bool(norm_ok and action_ok),
        }

        errors.append(row)

        print(
            "PARITY",
            json.dumps(row),
            flush=True
        )
    compiled_times=[]
    for repeat in range(2):
        for i,d in cases:
            ms=run(observations[i],noises[i],prefixes[i],d)[2]
            compiled_times.append(ms)
            print(f'COMPILED frame={i} delay={d} ms={ms:.1f}',flush=True)
    # Re-measure eager at the end to reduce warmup/cache/order bias.
    policy.model.denoise_step=eager_fn
    eager_times=[]
    for i,d in cases:
        eager_times.append(run(observations[i],noises[i],prefixes[i],d)[2])
    em=statistics.median(eager_times);cm=statistics.median(compiled_times)
    summary=dict(checkpoint=CHECKPOINT,compile_target='denoise_step',compile_mode='default',
                 parity_passed=all(x['passed'] for x in errors),errors=errors,
                 eager_median_ms=em,compiled_median_ms=cm,speedup=em/cm,
                 eager_samples_ms=eager_times,compiled_samples_ms=compiled_times,
                 integration_candidate=all(x['passed'] for x in errors) and cm<em*.9)
    output=Path('/root/autodl-tmp/pi05/logs/rtc_compile_benchmark_'+time.strftime('%Y%m%d_%H%M%S')+'.json')
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(summary,indent=2))
    print('=== RTC_COMPILE_RESULT ===',flush=True)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('errors','eager_samples_ms','compiled_samples_ms')},indent=2),flush=True)
    print('Report:',output,flush=True)

if __name__=='__main__':
    main()
