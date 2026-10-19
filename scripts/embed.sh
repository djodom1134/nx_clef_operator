#!/usr/bin/env bash
# Start (or restart) the EmbeddingGemma 2 server for the PTZ hunter in the background, next to Clef.
# Its Python is a venv that reuses Clef's CUDA torch (~/embed/.venv, see README). Logs to /tmp/embed_server.log.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -f /tmp/embed_server.pid ]] && kill -0 "$(cat /tmp/embed_server.pid)" 2>/dev/null; then
    kill "$(cat /tmp/embed_server.pid)"
    sleep 1
fi
HF_HOME="${HF_HOME:-$HOME/embed/hf}" HF_HUB_OFFLINE=1 setsid nohup "$HOME/embed/.venv/bin/python" -u scripts/embed_server.py \
    --port 8300 "$@" > /tmp/embed_server.log 2>&1 < /dev/null &
echo $! > /tmp/embed_server.pid
for _ in $(seq 1 60); do
    curl -s -m 2 http://127.0.0.1:8300/health && echo && exit 0
    sleep 2
done
echo "embedding server did not start; see /tmp/embed_server.log" >&2
exit 1
