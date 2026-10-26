"""LeRobot 0.6.1 guided RTC, using the installed implementation (not async-only)."""
import argparse
import base64
import hashlib
import io
import json
import time
import traceback
import uuid
from rtc_compile_integration import install_rtc_compile, rollback_rtc_compile
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.rtc.configuration_rtc import RTCConfig

TASK = 'Pick up the yellow cube from inside the black outlined area and place it in the gray-blue tray.'
DEFAULT_CHECKPOINT = '/root/autodl-tmp/pi05/outputs/pi05_recovery_cosine_tc_v1/checkpoints/002000/pretrained_model'
JOINTS = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']


def decode_image(encoded):
    with Image.open(io.BytesIO(base64.b64decode(encoded, validate=True))) as image:
        if image.size != (640, 480):
            raise ValueError('Expected RGB image 640x480')
        array = np.array(image.convert('RGB'), copy=True)
    return torch.from_numpy(array).permute(2, 0, 1).float() / 255


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default=DEFAULT_CHECKPOINT)
    ap.add_argument('--port', type=int, default=8088)
    ap.add_argument('--backend', choices=('eager', 'compiled'), default='eager')
    ap.add_argument('--allow-unverified-compile', action='store_true',
                    help='Explicitly use compiled outputs despite failed offline parity')
    args = ap.parse_args()
    if args.backend == 'compiled' and not args.allow_unverified_compile:
        ap.error('Compiled parity has not passed; use --allow-unverified-compile for experimental testing')
    checkpoint = str(Path(args.checkpoint).resolve())
    ready = False
    health = {}

    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, obj):
            data = json.dumps(obj, allow_nan=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self.reply(200 if self.path == '/health' else 404,
                       health if self.path == '/health' else {'error': 'Not found'})

        def do_POST(self):
            if self.path != '/infer':
                self.reply(404, {'error': 'Not found'})
                return
            try:
                begin = time.perf_counter()
                size = int(self.headers.get('Content-Length', 0))
                if not ready or not 0 < size <= 8_000_000:
                    raise ValueError('Not ready or invalid request size')
                self.connection.settimeout(5)
                request = json.loads(self.rfile.read(size))
                if request.get('protocol') != 'pi05-rtc-v1':
                    raise ValueError('Wrong protocol; use RTC client')
                if request.get('model_token') != health['model_token']:
                    raise ValueError('Server identity changed; reconnect explicitly')
                state = torch.tensor(request['state'], dtype=torch.float32)
                if state.shape != (6,) or not torch.isfinite(state).all():
                    raise ValueError('Expected six finite state values')
                obs = {'observation.images.front': decode_image(request['front']),
                       'observation.images.wrist': decode_image(request['wrist']),
                       'observation.state': state, 'task': TASK}
                prefix = request.get('prefix_normalized')
                delay = request.get('inference_delay', 0)
                if not isinstance(delay, int) or isinstance(delay, bool):
                    raise ValueError('inference_delay must be integer')
                if prefix is not None:
                    prefix = torch.tensor(prefix, device='cuda', dtype=torch.float32)
                    if prefix.ndim != 2 or prefix.shape[1] != 6 or not 1 <= len(prefix) <= 50:
                        raise ValueError('Expected normalized prefix [1..50,6]')
                    if not torch.isfinite(prefix).all() or not 0 <= delay < len(prefix):
                        raise ValueError('Invalid prefix or delay exceeds remaining actions')
                    prefix = prefix.unsqueeze(0)
                elif delay != 0:
                    raise ValueError('Bootstrap must have zero delay')
                try:
                    normalized, actions, ms = predict(obs, prefix, delay)
                except Exception as exc:
                    if compile_status.get('enabled'):
                        fallback(exc, 'request')
                    # Never retry an expired action request after a backend failure.
                    raise
                self.reply(200, dict(request_id=request['request_id'],
                                    model_token=health['model_token'],
                                    normalized_actions=normalized, actions=actions,
                                    rtc_guided=prefix is not None,
                                    inference_backend=health['inference_backend'],
                                    parity_verified=False,
                                    inference_ms=ms, server_ms=(time.perf_counter()-begin)*1000))
                print(f"request={request['request_id']} guided={prefix is not None} delay={delay} inference_ms={ms:.1f}", flush=True)
            except Exception as exc:
                traceback.print_exc()
                try:
                    self.reply(500, {'error': str(exc)})
                except (BrokenPipeError, ConnectionResetError):
                    pass

    # Bind before loading GPU weights; duplicate startup fails immediately.
    server = HTTPServer(('127.0.0.1', args.port), Handler)
    cfg = PI05Config.from_pretrained(checkpoint)
    cfg.pretrained_path = checkpoint
    cfg.device = 'cuda'
    cfg.gradient_checkpointing = False
    cfg.compile_model = False  # Compile only denoise_step below, never RTC guidance.
    cfg.num_inference_steps = 10
    cfg.rtc_config = RTCConfig(enabled=True, execution_horizon=30,
                               max_guidance_weight=10.0, debug=False)
    if cfg.chunk_size != 50 or cfg.use_relative_actions:
        raise RuntimeError('This client implements chunk=50 and absolute actions only')
    expected_names = [j + '.pos' for j in JOINTS]
    if cfg.action_feature_names is not None and list(cfg.action_feature_names) not in (JOINTS, expected_names):
        raise RuntimeError(f'Unexpected action order: {cfg.action_feature_names}')
    print('Loading checkpoint:', checkpoint, flush=True)
    policy = PI05Policy.from_pretrained(checkpoint, config=cfg, strict=True,
                                       local_files_only=True).eval()
    policy.init_rtc_processor()
    if not policy._rtc_enabled() or policy.model.rtc_processor is None:
        raise RuntimeError('RTC was not activated')
    pre, post = make_pre_post_processors(policy_cfg=cfg, pretrained_path=checkpoint)

    compile_status = dict(enabled=False, parity_verified=False)

    def sync_backend_status():
        health.update(compiled=bool(compile_status.get('enabled')),
                      inference_backend='compiled' if compile_status.get('enabled') else 'eager',
                      compile_status=dict(compile_status), parity_verified=False)

    def fallback(exc, phase):
        rollback_rtc_compile(policy)
        compile_status.update(enabled=False, parity_verified=False,
                              fallback_phase=phase,
                              fallback_reason=f'{type(exc).__name__}: {exc}')
        sync_backend_status()
        print('COMPILE_FALLBACK ' + json.dumps(compile_status), flush=True)

    try:
        compile_status.update(install_rtc_compile(
            policy, enabled=args.backend == 'compiled',
            allow_unverified=args.allow_unverified_compile))
    except Exception as exc:
        fallback(exc, 'install')
    sync_backend_status()

    @torch.no_grad()  # NOT inference_mode: RTC internally uses torch.enable_grad().
    def predict(obs, prefix=None, delay=0):
        torch.cuda.synchronize()
        start = time.perf_counter()
        batch = pre(obs)
        normalized = policy.predict_action_chunk(
            batch, prev_chunk_left_over=prefix, inference_delay=delay,
            execution_horizon=prefix.shape[1] if prefix is not None else 30,
        )
        # Preserve pre-postprocessor output; the queue must retain normalized actions.
        saved = normalized.detach().clone()
        physical = post(normalized.clone()).detach().to('cpu', copy=True)
        saved = saved.to('cpu', copy=True)
        torch.cuda.synchronize()
        for value in (saved, physical):
            if tuple(value.shape) != (1, 50, 6) or not torch.isfinite(value).all():
                raise RuntimeError('Invalid predicted actions')
        return saved[0].tolist(), physical[0].tolist(), (time.perf_counter()-start)*1000

    def warm_obs():
        return {'observation.images.front': torch.zeros(3, 480, 640),
                'observation.images.wrist': torch.zeros(3, 480, 640),
                'observation.state': torch.tensor([-12.8868,-21.2417,18.5084,79.0358,11.2939,3.4916]),
                'task': TASK}
    def warmup():
        normal, _, ms = predict(warm_obs())
        print(f'WARMUP bootstrap={ms:.1f}ms', flush=True)
        # Exercise bootstrap and several prefix lengths before accepting clients.
        for length, delay in ((30, 12), (29, 18), (35, 24), (30, 12)):
            prefix = torch.tensor(normal[-length:], dtype=torch.float32,
                                  device='cuda').unsqueeze(0)
            normal, _, ms = predict(warm_obs(), prefix, delay)
            print(f'WARMUP guided prefix={length} delay={delay} {ms:.1f}ms', flush=True)

    try:
        warmup()
    except Exception as exc:
        if not compile_status.get('enabled'):
            raise
        traceback.print_exc()
        fallback(exc, 'warmup')
        warmup()  # Failure here aborts startup; readiness is never reported.
    # Hash the saved processors so replies cannot cross model/normalization identities.
    digest = hashlib.sha256(checkpoint.encode())
    digest.update(uuid.uuid4().bytes)  # Reject queues from a prior server session.
    for p in sorted(Path(checkpoint).glob('policy_*processor*')):
        digest.update(p.read_bytes())
    health.update(ready=True, protocol='pi05-rtc-v1', rtc=True,
                  checkpoint=checkpoint, model_token=digest.hexdigest(), task=TASK,
                  chunk_size=50, action_dim=6, prefix_space='normalized',
                  joint_order=JOINTS, fps=30)
    sync_backend_status()
    ready = True
    print(f'SERVER_READY http://127.0.0.1:{args.port}', flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()

if __name__ == '__main__':
    main()
