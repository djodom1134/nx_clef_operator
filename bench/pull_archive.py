"""Pull recorded motion clips from the Nx archive for an evaluation set, with a contact sheet per clip for labelling.

Picks clips spread over the last few days (day and night) for each camera, downloads 30 s of the secondary stream
starting 3 s before each motion chunk, and writes ``<clip>.jpg``: six frames, one every 5 s, tiled 3 x 2.

    python3 -m bench.pull_archive --per-camera 4 --days 3
"""

from __future__ import annotations

import argparse
import json
import random
import ssl
import subprocess
import time
import urllib.request
from pathlib import Path

from clef_operator.env import ROOT
from clef_operator.nx import Nx, probe

_TLS = ssl.create_default_context()
_TLS.check_hostname = False
_TLS.verify_mode = ssl.CERT_NONE


def download(nx: Nx, camera_id: str, start_ms: int, seconds: int, out: Path) -> bool:
    request = urllib.request.Request(
        f"{nx.url}/media/{camera_id}.mp4?pos={start_ms}&duration={seconds}&stream=1",
        headers={"Authorization": f"Bearer {nx.token}"},
    )
    data = urllib.request.urlopen(request, timeout=120, context=_TLS).read()
    if len(data) < 10_000:
        return False
    out.write_bytes(data)
    return True


def contact_sheet(clip: Path) -> Path:
    sheet = clip.with_suffix(".jpg")
    subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(clip),
         "-vf", "fps=1/5,scale=426:-2,tile=3x2", "-frames:v", "1", "-q:v", "4", "-y", str(sheet)],
        check=True,
    )
    return sheet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-camera", type=int, default=4)
    parser.add_argument("--days", type=float, default=3)
    parser.add_argument("--seconds", type=int, default=30)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "clips" / "archive")
    args = parser.parse_args()

    random.seed(args.seed)
    nx = Nx()
    args.out.mkdir(parents=True, exist_ok=True)
    index = []
    since_ms = (time.time() - args.days * 86400) * 1000
    for cam in nx.cameras():
        chunks = [c for c in nx._request("GET", f"/rest/v2/devices/{cam.id}/footage")
                  if int(c["startTimeMs"]) > since_ms and int(c.get("durationMs", 0)) > 5000]
        night = [c for c in chunks if time.localtime(int(c["startTimeMs"]) / 1000).tm_hour in (21, 22, 23, 0, 1, 2, 3, 4, 5)]
        day = [c for c in chunks if c not in night]
        picks = random.sample(day, min(len(day), args.per_camera - args.per_camera // 4))
        picks += random.sample(night, min(len(night), args.per_camera // 4))
        for chunk in picks:
            start_ms = int(chunk["startTimeMs"]) - 3000
            stamp = time.strftime("%m%d-%H%M%S", time.localtime(start_ms / 1000))
            clip = args.out / f"{cam.name.strip().replace(' ', '_')}_{stamp}.mp4"
            if not clip.exists() and not download(nx, cam.id, start_ms, args.seconds, clip):
                print(f"skip {clip.name}: no video")
                continue
            contact_sheet(clip)
            info = probe(clip)
            index.append({"clip": clip.name, "camera": cam.name.strip(), "start_ms": start_ms, **info})
            print(f"{clip.name}: {info['width']}x{info['height']} {info['fps']:.1f} fps {info['duration']:.0f} s")
    (args.out / "index.json").write_text(json.dumps(index, indent=1))


if __name__ == "__main__":
    main()
