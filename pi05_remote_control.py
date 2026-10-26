import argparse
import base64
import io
import json
import time
from datetime import datetime
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image

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
    ap.add_argument("--url", default="http://127.0.0.1:18087")
    ap.add_argument("--port", default="/dev/ttyACM1")
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--execute-steps", type=int, default=10)
    ap.add_argument("--fps", type=float, default=30)
    ap.add_argument("--slew", type=float, default=5.0)
    ap.add_argument("--max-relative-target", type=float, default=10.0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not (args.seconds > 0 and args.fps > 0 and args.slew > 0
            and args.max_relative_target > 0
            and 1 <= args.execute_steps <= 50):
        raise ValueError("Invalid control settings")

    url = args.url.rstrip("/")
    health = request_json(url + "/health")
    if not health.get("ready") or health.get("chunk_size") != 50:
        raise RuntimeError(f"Server is not ready: {health}")
    if "pi05_recovery_cosine_tc_v1" not in health.get("checkpoint", ""):
        raise RuntimeError(f"Unexpected checkpoint: {health}")
    if health.get("task") != TASK:
        raise RuntimeError("Server task does not match training task")
    print("模型：", health["checkpoint"], flush=True)

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

    log_dir = Path(__file__).resolve().parent / "logs"
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / (
        "pi05_recovery_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".jsonl"
    )

    try:
        robot.connect()
        if set(robot.bus.motors) != set(JOINTS):
            raise RuntimeError(f"Unexpected motor names: {list(robot.bus.motors)}")

        with log_path.open("x", buffering=1) as log:
            log.write(json.dumps({
                "type": "config", "server": health, "args": vars(args),
                "joint_order": JOINTS,
            }) + "\n")

            print("已连接。Ctrl+C 停止；退出后保持电机力矩。", flush=True)
            print("日志：", log_path, flush=True)
            started = time.monotonic()
            deadline = started + args.seconds
            cycle = 0
            period = 1.0 / args.fps

            while time.monotonic() < deadline:
                obs = robot.get_observation()
                state = np.array([obs[k] for k in KEYS], dtype=np.float64)
                if not np.isfinite(state).all():
                    raise RuntimeError("Non-finite observed state")

                request_start = time.perf_counter()
                result = request_json(url + "/infer", {
                    "state": state.tolist(),
                    "front": encode_rgb(obs["front"]),
                    "wrist": encode_rgb(obs["wrist"]),
                })
                request_ms = (time.perf_counter() - request_start) * 1000
                actions = np.asarray(result["actions"], dtype=np.float64)
                if actions.shape != (50, 6) or not np.isfinite(actions).all():
                    raise RuntimeError(f"Invalid actions: {actions.shape}")

                cycle += 1
                print(
                    f"cycle={cycle} request={request_ms:.0f}ms "
                    f"model={result.get('inference_ms', 0):.0f}ms",
                    flush=True,
                )
                log.write(json.dumps({
                    "type": "prediction", "cycle": cycle,
                    "elapsed_s": time.monotonic() - started,
                    "state": state.tolist(), "actions": actions.tolist(),
                    "request_ms": request_ms,
                }) + "\n")

                if args.dry_run:
                    print("dry-run 完成，没有发送动作。", flush=True)
                    break

                # 以请求结束时的实际位置作为本段限速起点。
                present = robot.bus.sync_read("Present_Position")
                last = np.array([present[j] for j in JOINTS], dtype=np.float64)
                if not np.isfinite(last).all():
                    raise RuntimeError("Non-finite joint position")

                for target in actions[:args.execute_steps]:
                    if time.monotonic() >= deadline:
                        break
                    tick = time.monotonic()

                    # 当前 use_degrees=False：前五关节 [-100,100]，夹爪 [0,100]。
                    bounded = np.clip(
                        target, [-100, -100, -100, -100, -100, 0],
                        [100, 100, 100, 100, 100, 100],
                    )
                    command = last + np.clip(
                        bounded - last, -args.slew, args.slew
                    )
                    sent = robot.send_action(dict(zip(KEYS, command.tolist())))
                    last = np.array([sent[k] for k in KEYS], dtype=np.float64)
                    log.write(json.dumps({
                        "type": "action",
                        "elapsed_s": time.monotonic() - started,
                        "requested": target.tolist(),
                        "sent": last.tolist(),
                    }) + "\n")
                    time.sleep(max(0.0, period - (time.monotonic() - tick)))

    except KeyboardInterrupt:
        print("\n收到停止指令。", flush=True)
    finally:
        if robot.is_connected:
            try:
                present = robot.bus.sync_read("Present_Position")
                robot.send_action({f"{j}.pos": float(present[j]) for j in JOINTS})
                print("已发送当前位置保持指令。", flush=True)
            except Exception as exc:
                print(f"保持指令失败，请用硬件停止：{exc}", flush=True)
            finally:
                robot.disconnect()
        print("程序结束。", flush=True)

if __name__ == "__main__":
    main()
