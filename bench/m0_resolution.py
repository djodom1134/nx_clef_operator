"""M0 resolution test: does shrinking frames before Clef cut tokens and time, and does it change the answers?

Re-encodes existing clips at several widths (ffmpeg, H.264) and sends each to Clef at the same frame rates.

    python3 -m bench.m0_resolution results/clips/<stamp> --widths 640 448 320 224 --fps 2 6
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

from bench.m0_timing import QUESTIONS
from clef_operator.clef import Clef
from clef_operator.env import ROOT
from clef_operator.nx import probe


def scaled(path: Path, width: int) -> Path:
    out = path.with_name(f"{path.stem}_w{width}.mp4")
    if not out.exists():
        subprocess.run(
            ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(path), "-vf", f"scale={width}:-2",
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-an", "-y", str(out)],
            check=True,
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("clips", type=Path, help="a results/clips/<stamp> directory from m0_timing")
    parser.add_argument("--widths", type=int, nargs="+", default=[640, 448, 320, 224])
    parser.add_argument("--fps", type=float, nargs="+", default=[2, 6])
    parser.add_argument("--stream", default="s1", help="clip suffix to use: s1 (secondary) or s0 (primary)")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "m0_resolution.jsonl")
    args = parser.parse_args()

    clef = Clef()
    with args.out.open("a") as log:
        for clip in sorted(args.clips.glob(f"*_{args.stream}.mp4")):
            camera = clip.stem.rsplit("_", 1)[0].replace("_", " ")
            print(f"\n{camera}")
            for width in args.widths:
                path = scaled(clip, width)
                info = probe(path)
                for fps in args.fps:
                    if fps > info["fps"] + 0.5:
                        continue
                    result = clef.ask(f"Security camera '{camera}'.", QUESTIONS, videos=[path], video_fps=fps)
                    answers = {q: round(p["true"], 3) if "true" in p else {k: round(v, 3) for k, v in p.items()}
                               for q, p in result.answers.items()}
                    row = {"time": time.time(), "camera": camera, "width": info["width"], "height": info["height"],
                           "fps": fps, "tokens": result.input_tokens, "seconds": round(result.seconds, 3), "answers": answers}
                    log.write(json.dumps(row) + "\n")
                    print(f"  {info['width']:>4}x{info['height']:<4} {fps:>3g} fps: {result.input_tokens:>6} tokens "
                          f"{result.seconds:6.2f} s  person {answers['person']:.2f}  vehicle {answers['vehicle_moving']:.2f}")


if __name__ == "__main__":
    main()
