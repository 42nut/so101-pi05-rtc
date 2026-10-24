#!/usr/bin/env bash
set -euo pipefail
PI05_ENV=/root/autodl-tmp/pi05/env
PI05_LAUNCH_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
test -x "$PI05_ENV/bin/python"
test -f "$PI05_ENV/lib/libstdc++.so.6"
export LD_LIBRARY_PATH="$PI05_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LD_PRELOAD="$PI05_ENV/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
export TOKENIZERS_PARALLELISM=false
exec "$PI05_ENV/bin/python" -u "$PI05_LAUNCH_DIR/train_pi05_recovery.py" "$@"
