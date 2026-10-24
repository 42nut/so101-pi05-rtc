import base64
import io
import json
import time
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
from PIL import Image
import torch

from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.policies.factory import make_pre_post_processors

CHECKPOINT = (
    "/root/autodl-tmp/pi05/outputs/pi05_recovery_cosine_tc_v1/"
    "checkpoints/002000/pretrained_model"
)
TASK = (
    "Pick up the yellow cube from inside the black outlined area "
    "and place it in the gray-blue tray."
)

cfg = PI05Config.from_pretrained(CHECKPOINT)
cfg.pretrained_path = CHECKPOINT
cfg.device = "cuda"
cfg.compile_model = True
cfg.compile_mode = "reduce-overhead"
cfg.gradient_checkpointing = False
cfg.num_inference_steps = 10
assert cfg.chunk_size == 50, cfg.chunk_size

print("CHECKPOINT:", CHECKPOINT, flush=True)
print("TASK:", TASK, flush=True)
print("Loading policy...", flush=True)

policy = PI05Policy.from_pretrained(
    CHECKPOINT, config=cfg, strict=True, local_files_only=True
).eval()

pre, post = make_pre_post_processors(
    policy_cfg=cfg,
    pretrained_path=CHECKPOINT,
)

@torch.inference_mode()
def predict(obs):
    torch.compiler.cudagraph_mark_step_begin()
    torch.cuda.synchronize()
    start = time.perf_counter()
    batch = pre(obs)
    normalized = policy.predict_action_chunk(batch)
    actions = post(normalized).detach().to("cpu", copy=True)
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000

    if tuple(actions.shape) != (1, 50, 6):
        raise RuntimeError(f"Unexpected action shape: {actions.shape}")
    if not torch.isfinite(actions).all():
        raise RuntimeError("Non-finite actions")
    return actions[0].tolist(), elapsed_ms

def decode_image(encoded):
    raw = base64.b64decode(encoded, validate=True)
    with Image.open(io.BytesIO(raw)) as image:
        if image.size != (640, 480):
            raise ValueError(f"Expected 640x480 image, got {image.size}")
        array = np.array(image.convert("RGB"), copy=True)
    return torch.from_numpy(array).permute(2, 0, 1).float() / 255.0

class Handler(BaseHTTPRequestHandler):
    def reply(self, status, obj):
        data = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/health":
            self.reply(200, {
                "ready": True,
                "chunk_size": 50,
                "checkpoint": CHECKPOINT,
                "task": TASK,
            })
        else:
            self.reply(404, {"error": "Not found"})

    def do_POST(self):
        if self.path != "/infer":
            self.reply(404, {"error": "Not found"})
            return
        try:
            start = time.perf_counter()
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 8_000_000:
                raise ValueError("Invalid request size")

            request = json.loads(self.rfile.read(length))
            state = torch.tensor(request["state"], dtype=torch.float32)
            if state.shape != (6,) or not torch.isfinite(state).all():
                raise ValueError("Expected 6 finite state values")

            obs = {
                "observation.images.front": decode_image(request["front"]),
                "observation.images.wrist": decode_image(request["wrist"]),
                "observation.state": state,
                "task": TASK,
            }
            actions, inference_ms = predict(obs)
            server_ms = (time.perf_counter() - start) * 1000
            self.reply(200, {
                "actions": actions,
                "inference_ms": inference_ms,
                "server_ms": server_ms,
            })
            print(
                f"inference={inference_ms:.1f} ms "
                f"server={server_ms:.1f} ms",
                flush=True,
            )
        except Exception as exc:
            traceback.print_exc()
            self.reply(500, {"error": str(exc)})

# 提前占用端口，重复启动时立即报错，避免重复加载模型后的预热。
server = HTTPServer(("127.0.0.1", 8087), Handler)

print("Warming up; first call may take several minutes...", flush=True)
for i in range(5):
    obs = {
        "observation.images.front": torch.zeros(3, 480, 640),
        "observation.images.wrist": torch.zeros(3, 480, 640),
        "observation.state": torch.tensor(
            [-12.8868, -21.2417, 18.5084, 79.0358, 11.2939, 3.4916]
        ),
        "task": TASK,
    }
    _, ms = predict(obs)
    print(f"warmup {i + 1}/5: {ms:.1f} ms", flush=True)

print("SERVER_READY http://127.0.0.1:8087", flush=True)
server.serve_forever()
