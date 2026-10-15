#!/usr/bin/env bash
# Install the Clef Operator plugin into the local Nx Meta Server and restart it. Run on the Nx server with sudo:
#   sudo plugin/install.sh build/libclef_operator_plugin.so
set -euo pipefail

SO="${1:-build/libclef_operator_plugin.so}"
PLUGINS=/opt/networkoptix-metavms/mediaserver/bin/plugins
SERVICE=networkoptix-metavms-mediaserver

[[ -f "$SO" ]] || { echo "no plugin library at $SO" >&2; exit 1; }
[[ -d "$PLUGINS" ]] || { echo "no Nx Meta plugins directory at $PLUGINS" >&2; exit 1; }

install -d "$PLUGINS/clef_operator_plugin"
install -m 644 "$SO" "$PLUGINS/clef_operator_plugin/libclef_operator_plugin.so"
echo "installed to $PLUGINS/clef_operator_plugin/"
systemctl restart "$SERVICE"
echo "restarted $SERVICE; plugin log lines start with 'Clef Operator:' in the server log"
