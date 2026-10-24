#!/usr/bin/env bash
set -euo pipefail
PI05_ENV=/root/autodl-tmp/pi05/env
RTC_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export HF_HOME=/root/autodl-tmp/pi05/cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export LD_LIBRARY_PATH="$PI05_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LD_PRELOAD="$PI05_ENV/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
export CUDA_VISIBLE_DEVICES=0
exec "$PI05_ENV/bin/python" -u "$RTC_DIR/pi05_rtc_server.py" "$@"
