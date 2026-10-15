"""Offline replay (PRD: M1): cut a recording into 30 s windows every 20 s, send each through the gateway, and print
the events the state machine produces, with timings. The same windowing the plugin will do live.

    python3 -m clef_operator.replay clip.mp4 --gateway http://127.0.0.1:8200 --camera "Front porch" \\
        --question "fight:Is anyone fighting?:critical:8" --question "person:Is a person at the front door?:high:2"

A question is ``id:text:priority:fps``. Windows are replayed as fast as the gateway answers them.
"""

from __future__ import annotations

import argparse
import base64
import json
import subprocess
import tempfile
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

from .events import EventState, Question
from .nx import probe


def cut(path: Path, start: float, length: float, out: Path) -> bytes:
    subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-ss", f"{start:.3f}", "-i", str(path), "-t", f"{length:.3f}",
         "-c", "copy", "-an", "-y", str(out)],
        check=True,
    )
    return out.read_bytes()


def post(url: str, body: dict, token: str | None = None) -> dict:
    headers = {"Content-Type": "application/json", **({"Authorization": f"Bearer {token}"} if token else {})}
    request = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers)
    with urllib.request.urlopen(request, timeout=900) as response:
        return json.load(response)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("clip", type=Path)
    parser.add_argument("--gateway", default="http://127.0.0.1:8200")
    parser.add_argument("--token")
    parser.add_argument("--camera", default="Replay camera")
    parser.add_argument("--description", default="")
    parser.add_argument("--question", action="append", required=True, help="id:text:priority:fps")
    parser.add_argument("--window", type=float, default=30)
    parser.add_argument("--overlap", type=float, default=10)
    args = parser.parse_args()

    questions: dict[str, tuple[Question, str, float]] = {}
    for spec in args.question:
        qid, text, priority, fps = spec.split(":")
        questions[qid] = (Question(qid, text), priority, float(fps))
    groups: dict[tuple[str, float], list[str]] = defaultdict(list)
    for qid, (_, priority, fps) in questions.items():
        groups[(priority, fps)].append(qid)
    states = {qid: EventState(q) for qid, (q, _, _) in questions.items()}

    duration = probe(args.clip)["duration"]
    step = args.window - args.overlap
    base = time.time() - duration  # pretend the clip just finished recording, so window times look live
    with tempfile.TemporaryDirectory() as tmp:
        start = 0.0
        while start + args.window <= duration + 0.5:
            clip = cut(args.clip, start, args.window, Path(tmp) / "w.mp4")
            for (priority, fps), qids in groups.items():
                body = {
                    "camera_id": args.camera, "camera_name": args.camera, "camera_description": args.description,
                    "window_start": base + start, "window_end": base + start + args.window,
                    "priority": priority, "fps": fps, "motion": "active",
                    "questions": {qid: questions[qid][0].to_clef() for qid in qids},
                    "clip": base64.b64encode(clip).decode(),
                }
                sent = time.time()
                result = post(args.gateway.rstrip("/") + "/v1/jobs", body, args.token)
                if result["status"] != "answered":
                    print(f"[{start:6.1f}s] {priority} {fps:g} fps: {result['status']} {result.get('reason') or result.get('error')}")
                    continue
                line = []
                for qid in qids:
                    q = questions[qid][0]
                    p = q.alert_probability(result["answers"][qid])
                    line.append(f"{qid}={p:.2f}")
                    transition = states[qid].update(start, start + args.window, p)
                    if transition:
                        print(f"  >>> EVENT {transition.kind.upper()} {qid} at {transition.time:.0f}s (p={transition.probability:.2f})")
                print(f"[{start:6.1f}-{start + args.window:6.1f}s] {priority:<8} {fps:>3g} fps  {' '.join(line)}  "
                      f"model {result['model_seconds']:.2f}s queue {result['queue_seconds']:.2f}s "
                      f"round trip {time.time() - sent:.2f}s  {result['tokens']} tok")
            start += step
    for qid, state in states.items():
        if state.active:
            print(f"  >>> EVENT still open: {qid} since {state.started:.0f}s")


if __name__ == "__main__":
    main()
