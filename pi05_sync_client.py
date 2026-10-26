"""Synchronous no-prefix client for the existing pi05-rtc-v1 server.
Default read-only; --live enables writes. JPEG quality 95, 640x480.
"""
import argparse
import base64
import io
import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image
Image.init()  # Register codecs before starting the HTTP worker.

from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

JOINTS = [
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll", "gripper",
]
KEYS = [f"{name}.pos" for name in JOINTS]
TASK = (
    "Pick up the yellow cube from inside the black outlined area "
    "and place it in the gray-blue tray."
)

def request_json(url, payload=None, timeout=5):
    data = None if payload is None else json.dumps(payload).encode()
    req = Request(
        url, data=data,
        headers={"Content-Type": "application/json"},
    )
    with urlopen(req, timeout=timeout) as response:
        return json.load(response)

def encode_rgb(array):
    array = np.asarray(array)
    if array.shape != (480, 640, 3) or array.dtype != np.uint8:
        raise ValueError(f"Unexpected camera image: {array.shape}, {array.dtype}")
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="JPEG", quality=95)
    return base64.b64encode(buffer.getvalue()).decode("ascii")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--url', default='http://127.0.0.1:18088')
    ap.add_argument('--port', default='/dev/ttyACM1')
    ap.add_argument('--seconds', type=float, default=60)
    ap.add_argument('--fps', type=float, default=30)
    ap.add_argument('--execute-steps', type=int, default=30)
    ap.add_argument('--slew', type=float, default=10.0)
    ap.add_argument('--max-relative-target', type=float, default=20.0)
    ap.add_argument('--timeout', type=float, default=3.0)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument('--live', action='store_true', help='Send actions to robot')
    mode.add_argument('--dry-run', action='store_true', help='Read only (default)')
    args = ap.parse_args()
    args.dry_run = not args.live
    if not (np.isfinite([args.seconds, args.slew, args.max_relative_target, args.timeout]).all()
            and args.seconds > 0 and args.fps == 30 and 1 <= args.execute_steps <= 50
            and args.slew > 0 and args.max_relative_target > 0 and args.timeout > 0):
        raise ValueError('Invalid settings; fps must remain training fps=30')
    url = args.url.rstrip('/')
    health = request_json(url+'/health', timeout=args.timeout)
    expected = dict(protocol='pi05-rtc-v1', ready=True, rtc=True, chunk_size=50,
                    prefix_space='normalized', action_dim=6, task=TASK, joint_order=JOINTS, fps=30)
    for key, value in expected.items():
        if health.get(key) != value:
            raise RuntimeError(f'Server mismatch: {key}={health.get(key)}')
    if not health.get('checkpoint', '').rstrip('/').endswith('/pi05_dagger_334_v2/checkpoints/004000/pretrained_model'):
        raise RuntimeError('Unexpected checkpoint')
    print('SYNC model:', health['checkpoint'], flush=True)
    robot = SO101Follower(SO101FollowerConfig(
        port=args.port,
        id="my_awesome_follower_arm",
        use_degrees=False,
        disable_torque_on_disconnect=False,
        max_relative_target=args.max_relative_target,
        cameras={
            "front": OpenCVCameraConfig(
                index_or_path=Path(
                    "/dev/v4l/by-id/"
                    "usb-icSpring_icspring_camera_202404160005-video-index0"
                ),
                width=640, height=480, fps=30,
            ),
            "wrist": OpenCVCameraConfig(
                index_or_path=Path(
                    "/dev/v4l/by-id/"
                    "usb-icSpring_icspring_camera-video-index0"
                ),
                width=640, height=480, fps=30,
            ),
        },
    ))


    period = 1 / args.fps
    log_dir = Path(__file__).resolve().parent / 'logs'
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / ('pi05_sync_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.jsonl')
    log = log_path.open('x', buffering=1)
    started = time.monotonic()

    def emit(kind, **values):
        log.write(json.dumps(dict(type=kind, elapsed_s=time.monotonic()-started, **values), allow_nan=False)+'\n')

    def capture():
        obs = robot.get_observation()  # Only main thread touches hardware.
        state = np.asarray([obs[k] for k in KEYS], dtype=np.float64)
        if not np.isfinite(state).all():
            raise RuntimeError('Non-finite state')
        return state, np.array(obs['front'], copy=True), np.array(obs['wrist'], copy=True)

    def fetch(observation, prefix, delay, request_id):
        # Blocking request: no actions are consumed while waiting.
        state, front, wrist = observation
        result = request_json(url+'/infer', dict(
            protocol='pi05-rtc-v1', model_token=health['model_token'],
            request_id=request_id, state=state.tolist(),
            front=encode_rgb(front), wrist=encode_rgb(wrist),
            prefix_normalized=prefix, inference_delay=delay,
        ), timeout=args.timeout)
        if result.get('request_id') != request_id or result.get('model_token') != health['model_token']:
            raise RuntimeError('Stale or foreign server response')
        if result.get('rtc_guided') != (prefix is not None):
            raise RuntimeError('Server did not apply expected RTC mode')
        return result

    try:
        robot.connect()
        if set(robot.bus.motors) != set(JOINTS):
            raise RuntimeError('Unexpected motor names')
        emit('config', args=vars(args), server=health)
        print('Log:', log_path, flush=True)
        print('DRY RUN: no action writes' if args.dry_run else 'LIVE: Ctrl+C stops and holds', flush=True)

        deadline = time.monotonic() + args.seconds
        chunks = 0
        consumed = 0
        while time.monotonic() < deadline:
            observation = capture()
            rid = uuid.uuid4().hex
            begin = time.monotonic()
            result = fetch(observation, None, 0, rid)
            elapsed = time.monotonic() - begin
            actions = np.asarray(result['actions'], dtype=np.float64)
            if actions.shape != (50, 6) or not np.isfinite(actions).all():
                raise RuntimeError('Invalid action chunk')
            emit('prediction', request_id=rid, request_ms=elapsed*1000,
                 response=result, state=observation[0].tolist())
            if elapsed > args.timeout:
                raise RuntimeError('Total request duration exceeded timeout; discard reply')
            if time.monotonic() >= deadline:
                break
            # Refresh measured pose after waiting; never crop by network latency.
            present = robot.bus.sync_read('Present_Position')
            last = np.asarray([present[j] for j in JOINTS], dtype=np.float64)
            if not np.isfinite(last).all():
                raise RuntimeError('Invalid measured position')
            chunks += 1
            print(f'SYNC chunk={chunks} request={elapsed*1000:.0f}ms '
                  f'model={result["inference_ms"]:.0f}ms '
                  f'execute={args.execute_steps} rtc_guided=False', flush=True)
            next_tick = time.monotonic()
            for i, target in enumerate(actions[:args.execute_steps]):
                delay = next_tick - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                now = time.monotonic()
                if now >= deadline:
                    break
                bounded = np.clip(target, [-100]*5+[0], [100]*6)
                command = last + np.clip(bounded-last, -args.slew, args.slew)
                if args.dry_run:
                    sent = dict(zip(KEYS, command.tolist()))
                else:
                    sent = robot.send_action(dict(zip(KEYS, command.tolist())))
                last = np.asarray([sent[k] for k in KEYS], dtype=np.float64)
                if not np.isfinite(last).all():
                    raise RuntimeError('Non-finite sent action')
                consumed += 1
                emit('action', chunk=chunks, index=i, consumed=consumed,
                     requested=target.tolist(), sent=last.tolist(),
                     simulated=args.dry_run, tick_late_ms=max(0, now-next_tick)*1000)
                # Space writes by at least one period; never catch up in a burst.
                next_tick = time.monotonic() + period
            # Allow the final target one tick before observing again.
            delay = min(next_tick, deadline) - time.monotonic()
            if delay > 0:
                time.sleep(delay)
        print(f'Completed. SYNC chunks={chunks} actions={consumed}', flush=True)
    except KeyboardInterrupt:
        print('Stopped by user.', flush=True)
    except Exception as exc:
        emit('error', error=repr(exc))
        raise
    finally:
        if robot.is_connected:
            try:
                if not args.dry_run:
                    present = robot.bus.sync_read('Present_Position')
                    robot.send_action({f'{j}.pos': float(present[j]) for j in JOINTS})
                    print('Holding current pose; torque stays enabled.', flush=True)
            except Exception as exc:
                print(f'Hold failed: {exc}; use hardware stop.', flush=True)
            finally:
                robot.disconnect()
        log.close()

if __name__ == '__main__':
    main()
