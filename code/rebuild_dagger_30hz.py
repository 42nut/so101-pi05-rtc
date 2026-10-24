import copy
import json
import math
import shutil
from pathlib import Path

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from lerobot.datasets.lerobot_dataset import LeRobotDataset

BASE = Path("/root/autodl-tmp/pi05/datasets")
SOURCE = BASE / "raw/so101_dagger_upload/so101_dagger_v1"
OUTPUT = BASE / "work/so101_dagger_v1_30hz_v30"
AUDIT = BASE / "work/so101_dagger_v1_30hz_audit"
FPS = 30
VAL = [0, 18]
CAMERAS = [
    "observation.images.front",
    "observation.images.wrist",
]
KEYS = ["observation.state", "action", *CAMERAS]
TASK = (
    "Pick up the yellow cube from inside the black outlined area "
    "and place it in the gray-blue tray."
)


def read_json(path):
    return json.loads(path.read_text())


def read_source(i):
    folder = SOURCE / f"episode_{i:06d}"
    dataset = folder / "dataset"
    info = read_json(dataset / "meta/info.json")
    metadata = read_json(folder / "metadata.json")

    assert metadata["recorded_policy"] == "human_only", i
    assert metadata["status"] in ("accepted", "timing_review"), i
    assert info["total_episodes"] == 1, i

    rows = []
    with (folder / "rollout.jsonl").open() as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                if row.get("type") == "human_action":
                    rows.append(row)

    files = list((dataset / "data").rglob("*.parquet"))
    assert len(files) == 1, (i, "Parquet 数量异常")
    table = pq.read_table(files[0])
    n = table.num_rows

    assert n == len(rows) == info["total_frames"], i
    assert [r["frame"] for r in rows] == list(range(n)), i
    assert table["frame_index"].to_pylist() == list(range(n)), i

    times = np.array([r["t"] for r in rows], dtype=np.float64)
    times -= times[0]
    assert np.isfinite(times).all() and np.all(np.diff(times) > 0), i
    assert np.diff(times).max() < 0.2, (i, "存在长时间间隔")

    arrays = {}
    for key, log_key in [
        ("observation.state", "state"),
        ("action", "sent"),
    ]:
        saved = np.asarray(table[key].to_pylist(), dtype=np.float32)
        logged = np.asarray([r[log_key] for r in rows], dtype=np.float32)
        assert saved.shape == logged.shape == (n, 6), (i, key)
        assert np.isfinite(saved).all(), (i, key)
        np.testing.assert_allclose(saved, logged, rtol=1e-6, atol=1e-5)
        arrays[key] = saved

    # 末尾最多延长不足一帧，确保最终原始帧也被保留。
    grid = np.arange(math.ceil(times[-1] * FPS) + 1) / FPS
    source_index = np.searchsorted(times, grid, side="right") - 1
    source_index = np.clip(source_index, 0, n - 1)
    assert source_index[0] == 0 and source_index[-1] == n - 1

    video_paths = {}
    for key in CAMERAS:
        paths = [
            p for p in (dataset / "videos").rglob("*.mp4")
            if key in p.parts
        ]
        assert len(paths) == 1, (i, key, paths)
        video_paths[key] = paths[0]

    assert metadata["task"] == TASK, (i, "任务文本不一致")
    return {
        "id": i, "info": info, "arrays": arrays, "times": times,
        "grid": grid, "source_index": source_index,
        "videos": video_paths, "original_frames": n,
    }


class SequentialVideo:
    """按原始视频帧编号读取，重复索引复用同一图像。"""
    def __init__(self, path):
        self.container = av.open(str(path))
        self.decoder = self.container.decode(video=0)
        self.index = -1
        self.image = None

    def get(self, index):
        while self.index < index:
            try:
                frame = next(self.decoder)
            except StopIteration:
                raise RuntimeError(f"视频提前结束，需要原始帧 {index}")
            self.index += 1
            self.image = frame.to_ndarray(format="rgb24")
            assert self.image.shape == (480, 640, 3)
        return self.image

    def check_end(self):
        try:
            next(self.decoder)
        except StopIteration:
            return
        raise RuntimeError("视频帧数超过对应的动作表")

    def close(self):
        self.container.close()


def main():
    if OUTPUT.exists() or AUDIT.exists():
        raise RuntimeError(
            f"输出或审计目录已存在，未覆盖：\n{OUTPUT}\n{AUDIT}"
        )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    # 为临时 PNG、视频和输出预留空间。
    free_gib = shutil.disk_usage(OUTPUT.parent).free / 1024**3
    if free_gib < 30:
        raise RuntimeError(f"可用空间 {free_gib:.1f} GiB，请先保证至少 30 GiB")

    print("检查全部原始记录...", flush=True)
    episodes = [read_source(i) for i in range(20)]

    features = {
        key: copy.deepcopy(episodes[0]["info"]["features"][key])
        for key in KEYS
    }
    for key in KEYS:
        features[key].pop("info", None)
        features[key]["shape"] = tuple(features[key]["shape"])
        for ep in episodes:
            other = copy.deepcopy(ep["info"]["features"][key])
            other.pop("info", None)
            other["shape"] = tuple(other["shape"])
            assert other == features[key], (ep["id"], key, "特征不一致")

    AUDIT.mkdir()
    (AUDIT / "status.json").write_text(json.dumps({"status": "building"}))

    ds = LeRobotDataset.create(
        repo_id="local/so101_dagger_v1_30hz",
        root=OUTPUT,
        fps=FPS,
        features=features,
        robot_type=episodes[0]["info"].get("robot_type") or "so101_follower",
        use_videos=True,
        image_writer_processes=0,
        image_writer_threads=4,
        batch_encoding_size=1,
        video_backend="pyav",
    )

    records = []
    for ep in episodes:
        i = ep["id"]
        readers = {
            key: SequentialVideo(path)
            for key, path in ep["videos"].items()
        }
        print(
            f"REBUILD {i:02d}: {ep['original_frames']} -> "
            f"{len(ep['grid'])} 帧，实际跨度 {ep['times'][-1]:.3f}s",
            flush=True,
        )
        try:
            for src_idx in ep["source_index"]:
                j = int(src_idx)
                frame = {
                    "observation.state": ep["arrays"]["observation.state"][j].copy(),
                    "action": ep["arrays"]["action"][j].copy(),
                    "task": TASK,
                }
                for key in CAMERAS:
                    frame[key] = readers[key].get(j).copy()
                ds.add_frame(frame)

            for reader in readers.values():
                reader.check_end()
        finally:
            for reader in readers.values():
                reader.close()

        ds.save_episode()

        np.savez_compressed(
            AUDIT / f"episode_{i:06d}_mapping.npz",
            source_time_s=ep["times"],
            output_time_s=ep["grid"],
            source_frame_index=ep["source_index"],
        )
        records.append({
            "collection_episode": i,
            "output_episode": i,
            "split": "val" if i in VAL else "train",
            "original_frames": ep["original_frames"],
            "output_frames": len(ep["grid"]),
            "real_span_s": float(ep["times"][-1]),
            "output_span_s": float(ep["grid"][-1]),
        })
        print(f"SAVED {i:02d}", flush=True)

    ds.finalize()
    print("写入完成，检查输出动作表...", flush=True)

    info = read_json(OUTPUT / "meta/info.json")
    expected_total = sum(len(ep["grid"]) for ep in episodes)
    assert info["codebase_version"] == "v3.0", info["codebase_version"]
    assert info["total_episodes"] == 20
    assert info["total_frames"] == expected_total
    assert info["fps"] == FPS

    files = sorted((OUTPUT / "data").rglob("*.parquet"))
    table = pa.concat_tables([pq.read_table(p) for p in files])
    table = table.sort_by([("index", "ascending")])
    assert table.num_rows == expected_total
    assert table["index"].to_pylist() == list(range(expected_total))

    episode_ids = np.array(table["episode_index"].to_pylist())
    for ep in episodes:
        i = ep["id"]
        part = table.filter(pa.array(episode_ids == i))
        n = len(ep["grid"])
        assert part.num_rows == n
        assert part["frame_index"].to_pylist() == list(range(n))
        np.testing.assert_allclose(
            part["timestamp"].to_pylist(), ep["grid"],
            rtol=0, atol=1e-4,
        )
        for key in ("observation.state", "action"):
            actual = np.asarray(part[key].to_pylist(), dtype=np.float32)
            expected = ep["arrays"][key][ep["source_index"]]
            np.testing.assert_array_equal(actual, expected)

    print("检查全部输出视频的解码和帧数...", flush=True)
    for key in CAMERAS:
        paths = [
            p for p in (OUTPUT / "videos").rglob("*.mp4")
            if key in p.parts
        ]
        assert paths, (key, "缺少视频")
        count = 0
        for path in sorted(paths):
            with av.open(str(path)) as container:
                for frame in container.decode(video=0):
                    assert (frame.width, frame.height) == (640, 480)
                    count += 1
        assert count == expected_total, (key, count, expected_total)
        print(f"VIDEO_OK {key}: {count} 帧", flush=True)

    manifest = {
        "dataset_root": str(OUTPUT),
        "source_root": str(SOURCE),
        "fps": FPS,
        "resampling": "previous_complete_observation_action_tuple",
        "time_source": "human_action log t; not camera exposure timestamps",
        "train_episodes": [i for i in range(20) if i not in VAL],
        "val_episodes": VAL,
        "total_episodes": 20,
        "total_frames": expected_total,
        "episodes": records,
    }
    (AUDIT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    (AUDIT / "status.json").write_text(
        json.dumps({"status": "complete", "checks_passed": True}) + "\n"
    )

    print("\nREBUILD_COMPLETE", flush=True)
    print("数据：", OUTPUT, flush=True)
    print("清单：", AUDIT / "manifest.json", flush=True)
    print(f"20 条，{expected_total} 帧，30 FPS，v3.0", flush=True)
    print("训练 18 条；验证 [0, 18]", flush=True)


if __name__ == "__main__":
    main()
