"""Async RTC with stationary rebootstrap recovery. Default read-only.
Existing pi05-rtc-v1 server, checkpoint dagger 004000.
JPEG 95, 640x480, 30Hz. Expired replies never reused.
"""
import argparse
import base64
import io
import json
import time
import math
import uuid
from concurrent.futures import ThreadPoolExecutor
from collections import deque
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

class StaleReply(RuntimeError):
    pass


def checked_chunk(value):
    result = np.asarray(value, dtype=np.float32)
    if result.shape != (50, 6) or not np.isfinite(result).all():
        raise ValueError(f"Invalid action chunk: {result.shape}")
    return result.copy()


class Queue:
    def __init__(self):
        self.normalized = None
        self.actions = None
        self.cursor = 0
        self.consumed = 0
        self.generation = 0

    def remaining(self):
        return 0 if self.actions is None else len(self.actions) - self.cursor

    def bootstrap(self, normalized, actions):
        self.normalized = checked_chunk(normalized)
        self.actions = checked_chunk(actions)
        self.cursor = 0
        self.generation += 1

    def snapshot(self):
        if self.remaining() < 1:
            raise StaleReply("Empty queue: cannot build an RTC prefix")
        return dict(generation=self.generation, consumed=self.consumed,
                    remaining=self.remaining(),
                    prefix=self.normalized[self.cursor:].copy().tolist())

    def merge(self, snapshot, normalized, actions):
        delay = self.consumed - snapshot['consumed']
        if snapshot['generation'] != self.generation or not 0 <= delay < snapshot['remaining']:
            raise StaleReply("Reply belongs to an old/exhausted action timeline")
        norm, physical = checked_chunk(normalized), checked_chunk(actions)
        if delay >= len(physical):
            raise StaleReply("Entire reply has expired")
        self.normalized, self.actions = norm, physical
        self.cursor = delay  # Do not replay actions elapsed during inference.
        self.generation += 1
        return delay

    def pop(self):
        if self.remaining() < 1:
            raise StaleReply("Queue underrun: stop instead of replaying old actions")
        action = self.actions[self.cursor].copy()
        self.cursor += 1
        self.consumed += 1
        return action


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
    ap.add_argument('--request-after', type=int, default=1,
                    help='Request when current chunk cursor reaches this index')
    ap.add_argument('--initial-delay', type=int, default=12)
    ap.add_argument('--guidance', choices=('rtc', 'none'), default='rtc',
                    help='Only change server prefix guidance; retain async timing/cropping')
    ap.add_argument('--slew', type=float, default=10.0)
    ap.add_argument('--max-relative-target', type=float, default=20.0)
    ap.add_argument('--timeout', type=float, default=3.0)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument('--live', action='store_true', help='Enable motor action writes')
    mode.add_argument('--dry-run', action='store_true', help='Read-only; default')
    ap.add_argument('--max-recoveries', type=int, default=10)
    ap.add_argument('--settle-seconds', type=float, default=0.1)
    args = ap.parse_args()
    args.dry_run = not args.live
    if not (np.isfinite([args.seconds, args.timeout, args.slew, args.max_relative_target, args.settle_seconds]).all()
            and args.max_recoveries >= 0 and 0 <= args.settle_seconds <= 2 and args.seconds > 0 and args.fps == 30 and 1 <= args.request_after <= 35
            and 1 <= args.initial_delay < 50-args.request_after
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
    print('RTC model:', health['checkpoint'], flush=True)
    print('ASYNC guidance=' + args.guidance + '; actual-delay cropping enabled', flush=True)
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


    queue = Queue()
    latencies = deque(maxlen=20)
    pool = ThreadPoolExecutor(max_workers=1)
    pending = None
    last = None
    period = 1 / args.fps
    log_dir = Path(__file__).resolve().parent / 'logs'
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / ('pi05_async_ab_' + args.guidance + '_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.jsonl')
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
        # Worker does ONLY encoding/HTTP; no robot, queue or log access.
        state, front, wrist = observation
        wire_prefix = prefix if args.guidance == 'rtc' else None
        wire_delay = delay if wire_prefix is not None else 0
        fetch_started = time.monotonic()
        result = request_json(url+'/infer', dict(
            protocol='pi05-rtc-v1', model_token=health['model_token'],
            request_id=request_id, state=state.tolist(),
            front=encode_rgb(front), wrist=encode_rgb(wrist),
            prefix_normalized=wire_prefix, inference_delay=wire_delay,
        ), timeout=args.timeout)
        if time.monotonic() - fetch_started > args.timeout:
            raise TimeoutError('Total inference request exceeded timeout')
        checked_chunk(result['actions'])
        checked_chunk(result['normalized_actions'])
        if result.get('request_id') != request_id or result.get('model_token') != health['model_token']:
            raise RuntimeError('Stale or foreign server response')
        if result.get('rtc_guided') != (wire_prefix is not None):
            raise RuntimeError('Server did not apply expected RTC mode')
        return result

    recoveries = 0
    deadline = float('inf')

    def hold():
        nonlocal last
        present = robot.bus.sync_read('Present_Position')
        last = np.asarray([present[j] for j in JOINTS], dtype=np.float64)
        if not np.isfinite(last).all():
            raise RuntimeError('Invalid position during hold')
        if not args.dry_run:
            sent = robot.send_action(dict(zip(KEYS, last.tolist())))
            last = np.asarray([sent[k] for k in KEYS], dtype=np.float64)
            if not np.isfinite(last).all():
                raise RuntimeError('Invalid hold result')

    def recover(reason):
        nonlocal pending, recoveries
        hold()
        queue.bootstrap(np.zeros((50, 6)), np.zeros((50, 6)))
        queue.cursor = 50  # Invalidate old generation and leave no executable actions.
        emit('recovery_start', reason=reason, recovery=recoveries+1)
        print(f'RECOVERY reason={reason}: holding, discard old plan', flush=True)
        if recoveries >= args.max_recoveries:
            raise RuntimeError('Recovery limit reached; stopped while holding')
        recoveries += 1
        if pending is not None:
            old_future, old_snap, old_time, old_delay, old_id = pending
            # Drain before taking a fresh observation: the server is single-threaded.
            # Transport errors, foreign replies and malformed actions remain fatal.
            old_future.result(timeout=args.timeout + 1)
            emit('discard_reply', request_id=old_id)
            pending = None
        if time.monotonic() >= deadline:
            return False
        time.sleep(min(args.settle_seconds, max(0, deadline-time.monotonic())))
        if time.monotonic() >= deadline:
            return False
        observation = capture()
        begin = time.monotonic()
        result = fetch(observation, None, 0, uuid.uuid4().hex)
        elapsed = time.monotonic()-begin
        if time.monotonic() >= deadline:
            return False
        queue.bootstrap(result['normalized_actions'], result['actions'])
        hold()
        # Restart estimator for the new stationary bootstrap timeline.
        latencies.clear()
        latencies.append(elapsed)
        emit('recovery_ready', recovery=recoveries, request_ms=elapsed*1000,
             response=result)
        print(f'RECOVERY ready={recoveries} request={elapsed*1000:.0f}ms; resume RTC', flush=True)
        return True

    try:
        robot.connect()
        if set(robot.bus.motors) != set(JOINTS):
            raise RuntimeError('Unexpected motor names')
        emit('config', args=vars(args), server=health)
        print('Log:', log_path, flush=True)
        print('DRY RUN: no action writes' if args.dry_run else 'LIVE: Ctrl+C stops and holds', flush=True)

        observation = capture()
        first_start = time.monotonic()
        # Bootstrap: robot is stationary, so no elapsed action rows are discarded.
        first = fetch(observation, None, 0, uuid.uuid4().hex)
        queue.bootstrap(first['normalized_actions'], first['actions'])
        latencies.append(time.monotonic()-first_start)
        present = robot.bus.sync_read('Present_Position')
        last = np.asarray([present[j] for j in JOINTS], dtype=np.float64)
        emit('bootstrap', response=first)
        deadline = time.monotonic() + args.seconds
        next_tick = time.monotonic()
        merged = 0

        while time.monotonic() < deadline:
            now = time.monotonic()
            if pending is not None:
                future, snap, requested_at, predicted_delay, rid = pending
                age = now-requested_at
                # A stalled controller must not silently make an old prediction fresh.
                if age >= snap['remaining']*period:
                    if not recover('response_deadline'):
                        break
                    next_tick = time.monotonic()
                    continue
                if future.done():
                    result = future.result()  # Any HTTP failure stops action production.
                    delay = queue.merge(snap, result['normalized_actions'], result['actions'])
                    latencies.append(age)
                    emit('merge', request_id=rid, actual_delay_steps=delay,
                         estimated_delay_steps=predicted_delay, request_ms=age*1000,
                         response=result, remaining=queue.remaining())
                    print(f'RTC merge={merged+1} actual_delay={delay} predicted={predicted_delay} '
                          f'request={age*1000:.0f}ms model={result["inference_ms"]:.0f}ms '
                          f'queue={queue.remaining()}', flush=True)
                    merged += 1
                    pending = None

            if pending is None and queue.cursor >= args.request_after:
                predicted_delay = max(args.initial_delay,
                                      math.ceil(max(latencies)*args.fps)+2)
                if predicted_delay >= queue.remaining():
                    if not recover('latency_budget'):
                        break
                    next_tick = time.monotonic()
                    continue
                observation = capture()
                snap = queue.snapshot()
                rid = uuid.uuid4().hex
                requested_at = time.monotonic()
                future = pool.submit(fetch, observation, snap['prefix'], predicted_delay, rid)
                pending = (future, snap, requested_at, predicted_delay, rid)
                emit('request', request_id=rid, state=observation[0].tolist(),
                     snapshot_consumed=snap['consumed'], remaining=snap['remaining'],
                     predicted_delay=predicted_delay)

            now = time.monotonic()
            if now < next_tick:
                time.sleep(min(next_tick-now, .005))
                continue
            if queue.remaining() == 0:
                if not recover('queue_empty'):
                    break
                next_tick = time.monotonic()
                continue
            target = queue.pop()
            bounded = np.clip(target, [-100]*5+[0], [100]*6)
            command = last+np.clip(bounded-last, -args.slew, args.slew)
            if args.dry_run:
                sent = dict(zip(KEYS, command.tolist()))
            else:
                sent = robot.send_action(dict(zip(KEYS, command.tolist())))
            last = np.asarray([sent[k] for k in KEYS], dtype=np.float64)
            if not np.isfinite(last).all():
                raise RuntimeError('Non-finite sent action')
            emit('action', consumed=queue.consumed, requested=target.tolist(),
                 sent=last.tolist(), simulated=args.dry_run,
                 remaining=queue.remaining(), tick_late_ms=max(0, now-next_tick)*1000)
            # No catch-up burst after slow reads or scheduling stalls.
            next_tick = max(next_tick+period, time.monotonic())
        print(f'Completed. RTC merges={merged} recoveries={recoveries}', flush=True)
    except KeyboardInterrupt:
        print('Stopped by user.', flush=True)
    except Exception as exc:
        emit('error', error=repr(exc))
        raise
    finally:
        if pending is not None:
            pending[0].cancel()
        pool.shutdown(wait=False, cancel_futures=True)
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
