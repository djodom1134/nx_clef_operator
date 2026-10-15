#!/bin/bash
# Clef Operator: copy the plugin into the Server's plugins folder on every container start (the folder is not a
# volume, so a recreated container would otherwise lose it). Installed by the nx_clef_operator session.
src=/opt/mediaserver/entrypoint.d/clef_operator/libclef_operator_plugin.so
dst=/opt/networkoptix-metavms/mediaserver/bin/plugins/clef_operator_plugin
if [[ -f "$src" ]]; then
    mkdir -p "$dst" && cp -f "$src" "$dst/" && echo "Clef Operator plugin installed in $dst"
fi
