#!/usr/bin/env bash
# Start (or restart) the Clef gateway in the background. Usage: scripts/gateway.sh [extra gateway args]
# Logs to /tmp/clefop_gateway.log.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/bin:$PATH"  # a static ffmpeg in ~/bin on machines without one (the DGX)
if [[ -f /tmp/clefop_gateway.pid ]] && kill -0 "$(cat /tmp/clefop_gateway.pid)" 2>/dev/null; then
    kill "$(cat /tmp/clefop_gateway.pid)"
    sleep 1
fi
nohup python3 -u -m clef_operator.gateway --port 8200 "$@" > /tmp/clefop_gateway.log 2>&1 < /dev/null &
echo $! > /tmp/clefop_gateway.pid
sleep 2
curl -s -m 10 http://127.0.0.1:8200/health; echo
