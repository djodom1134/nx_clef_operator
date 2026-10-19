"""Add each Clef camera's status strip to Nx: a Web Page per camera (under Integrations) and a shared layout
"Clef status" with each camera beside its strip, two pairs to a row. Run where the gateway and Nx are reachable,
with the Nx login in the environment (.env). Pages and the layout that already exist are left alone.

    python3 scripts/nx_strips.py --gateway http://192.168.87.134:8200
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from clef_operator.nx import Nx  # noqa: E402

LAYOUT = "Clef status"
SHARED = "00000000-0000-0000-0000-000000000000"  # a local admin may not create layouts owned by another user


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gateway", required=True, help="the gateway URL as Nx clients reach it")
    parser.add_argument("--token", default="", help="the gateway token, if it has one")
    args = parser.parse_args()
    gateway = args.gateway.rstrip("/")
    token = f"&token={args.token}" if args.token else ""
    nx = Nx()
    with urllib.request.urlopen(f"{gateway}/v1/cameras?{token.lstrip('&')}", timeout=30) as response:
        cameras = [c for c in json.load(response)["cameras"] if c["has_answers"]]
    pages = {p["name"]: p for p in nx._request("GET", "/rest/v4/webPages")}
    strips = {}
    for camera in cameras:
        name = f"Clef · {' '.join(camera['name'].split())}"
        if name not in pages:
            pages[name] = nx._request("POST", "/rest/v4/webPages", {
                "name": name, "url": f"{gateway}/strip?camera={camera['id']}{token}",
                "parameters": {"subtype": "clientApi"}, "proxyDomainAllowList": [], "certificateCheck": False})
            print(f"web page {name}")
        strips[camera["id"]] = pages[name]["id"]
    if any(layout["name"] == LAYOUT for layout in nx._request("GET", "/rest/v4/layouts")):
        print(f"layout '{LAYOUT}' exists")
        return
    item = {"flags": 1, "rotation": 0, "resourcePath": "", "zoomLeft": 0, "zoomTop": 0, "zoomRight": 0, "zoomBottom": 0,
            "zoomTargetId": SHARED, "displayInfo": False, "displayAnalyticsObjects": False, "displayRoi": True,
            "controlPtz": False, "displayHotspots": True,
            "contrastParams": {"enabled": False, "blackLevel": 0.001, "whiteLevel": 0.0005, "gamma": 1},
            "dewarpingParams": {"enabled": False, "xAngle": 0, "yAngle": 0, "fov": 1.2217304763960306, "panoFactor": 1}}
    items = []
    for i, camera in enumerate(cameras):
        col, row = (i % 2) * 2, i // 2
        for dx, resource in ((0, camera["id"]), (1, strips[camera["id"]])):
            items.append({**item, "id": str(uuid.uuid4()), "resourceId": resource,
                          "left": col + dx, "top": row, "right": col + dx + 1, "bottom": row + 1})
    nx._request("POST", "/rest/v4/layouts", {
        "name": LAYOUT, "parentId": SHARED, "cellAspectRatio": 16 / 9, "cellSpacing": 0.05, "locked": False,
        "backgroundImageFilename": "", "backgroundWidth": -1, "backgroundHeight": -1, "backgroundOpacity": 0.7,
        "fixedWidth": 0, "fixedHeight": 0, "logicalId": 0, "items": items})
    print(f"layout '{LAYOUT}': {len(cameras)} cameras")


if __name__ == "__main__":
    main()
