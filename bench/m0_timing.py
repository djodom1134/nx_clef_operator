"""M0 timing benchmark: how long Clef takes on a 30 s window at each frame rate.

Captures live 30 s clips (secondary and primary stream) from Nx cameras, then sends each clip to the Clef server at
each frame rate with the same small question set, and records tokens and seconds. Run it on a machine with ffmpeg
that can reach both the Nx server and the Clef server, e.g. the Nx server itself:

    python3 -m bench.m0_timing --cameras 3 --fps 1 2 4 6 8 10
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from clef_operator.clef import Clef, noul, score
from clef_operator.env import ROOT
from clef_operator.nx import Nx, probe

QUESTIONS = {
    "person": noul("Is there at least one person visible at any point in this clip?"),
    "vehicle_moving": noul("Is a vehicle moving at any point in this clip?"),
    "fighting": noul("Are two or more people fighting or physically struggling with each other?"),
    "people_count": score("How many people are visible at the busiest moment of this clip?", ["none", "one", "two or three", "four or more"]),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cameras", type=int, default=3, help="how many recording cameras to use")
    parser.add_argument("--names", nargs="*", help="camera names to use instead of the first N")
    parser.add_argument("--seconds", type=float, default=30)
    parser.add_argument("--fps", type=float, nargs="+", default=[1, 2, 4, 6, 8, 10])
    parser.add_argument("--streams", type=int, nargs="+", default=[1, 0], help="1 = secondary, 0 = primary")
    parser.add_argument("--repeat", type=int, default=1, help="extra timed runs per (clip, fps)")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "m0_timing.jsonl")
    args = parser.parse_args()

    nx, clef = Nx(), Clef()
    print("clef:", clef.health())
    cameras = nx.cameras()
    cameras = [c for c in cameras if c.name in args.names] if args.names else cameras[: args.cameras]
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    clip_dir = ROOT / "results" / "clips" / stamp

    def capture(job):
        cam, stream = job
        try:
            return nx.capture_live(cam.id, args.seconds, clip_dir / f"{cam.name.replace(' ', '_')}_s{stream}.mp4", stream)
        except Exception as error:  # noqa: BLE001 - e.g. a camera without a secondary stream
            print(f"capture failed: {cam.name} stream {stream}: {error}")
            return None

    jobs = [(cam, stream) for cam in cameras for stream in args.streams]
    print(f"capturing {len(jobs)} clips of {args.seconds:.0f} s in parallel ...")
    with ThreadPoolExecutor(len(jobs)) as pool:
        captured = list(pool.map(capture, jobs))
    jobs, paths = [j for j, p in zip(jobs, captured) if p], [p for p in captured if p]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with args.out.open("a") as log:
        for (cam, stream), path in zip(jobs, paths):
            info = probe(path)
            print(f"\n{cam.name} stream {stream}: {info['codec']} {info['width']}x{info['height']} "
                  f"{info['fps']:.1f} fps, {info['bytes'] / 1e6:.1f} MB")
            state = f"Security camera '{cam.name}'. Local time {datetime.now():%A %H:%M}."
            for fps in args.fps:
                if fps > info["fps"] + 0.5:
                    print(f"  {fps:>4g} fps: skipped, source has {info['fps']:.1f} fps")
                    continue
                for run in range(1 + args.repeat):
                    try:
                        result = clef.ask(state, QUESTIONS, videos=[path], video_fps=fps)
                    except Exception as error:  # noqa: BLE001 - record the failure and keep going
                        print(f"  {fps:>4g} fps: ERROR {error}")
                        break
                    row = {
                        "time": time.time(), "camera": cam.name, "stream": stream, "codec": info["codec"],
                        "width": info["width"], "height": info["height"], "source_fps": round(info["fps"], 2),
                        "clip_mb": round(info["bytes"] / 1e6, 2), "fps": fps, "run": run,
                        "tokens": result.input_tokens, "seconds": round(result.seconds, 3),
                        "answers": {q: round(p.get("true", max(p.values())), 3) if "true" in p else p for q, p in result.answers.items()},
                    }
                    rows.append(row)
                    log.write(json.dumps(row) + "\n")
                    print(f"  {fps:>4g} fps run {run}: {result.input_tokens:>6} tokens  {result.seconds:6.2f} s  "
                          f"person {result.answers['person']['true']:.2f}  fighting {result.answers['fighting']['true']:.2f}")

    print("\nsummary (median over cameras and runs)")
    print(f"{'stream':>6} {'fps':>4} {'tokens':>7} {'seconds':>8} {'n':>3}")
    for stream in args.streams:
        for fps in args.fps:
            sel = [r for r in rows if r["stream"] == stream and r["fps"] == fps]
            if sel:
                print(f"{stream:>6} {fps:>4g} {statistics.median(r['tokens'] for r in sel):>7.0f} "
                      f"{statistics.median(r['seconds'] for r in sel):>8.2f} {len(sel):>3}")


if __name__ == "__main__":
    main()
