#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LOGDIR=/root/autodl-tmp/pi05/logs
mkdir -p "$LOGDIR"
# Lock is inherited by the training process; preflight and training cannot overlap.
exec 9>"$HERE/.dagger_training.lock"
flock -n 9 || { echo '本目录已有训练或预检进程，未重复启动。'; exit 1; }
CUDA_VISIBLE_DEVICES=0 bash "$HERE/run_pi05_dagger.sh" --check-only "$@"
LOGFILE="$LOGDIR/pi05_dagger_334_$(date +%Y%m%d_%H%M%S).log"
nohup env CUDA_VISIBLE_DEVICES=0 bash "$HERE/run_pi05_dagger.sh" "$@" > "$LOGFILE" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$PID" > "$HERE/training.pid"
printf '后台训练 PID：%s\n日志：%s\n' "$PID" "$LOGFILE"
