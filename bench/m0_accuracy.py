"""M0 accuracy: Clef on hand-labelled archive clips (bench/labels.json) at several frame rates and widths.

Asks a person question and the nuisance-source choice question (PRD: Motion and nuisance learning) and reports
person accuracy (threshold 0.5), the probability gap between clips with and without people, and nuisance top-1.

    python3 -m bench.m0_accuracy results/clips/archive --configs 2:native 2:448 1:448 4:448
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from clef_operator.clef import Clef, choice, noul
from clef_operator.env import ROOT
from clef_operator.gateway import prepare_clip

NUISANCE = [
    "trees or plants moving in wind", "a flag or banner waving", "shadows moving",
    "lights turning on or off, or changing light", "vehicle headlights", "rain, snow or fog",
    "insects or a spider web near the lens", "camera shake", "water or reflections",
]
REAL = ["a person", "a vehicle", "an animal", "nothing visible"]
QUESTIONS = {
    "person": noul("Is there at least one person visible at any point in this clip?"),
    "cause": choice("What is causing the movement in this clip?", {o: o for o in NUISANCE + REAL}),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("clips", type=Path)
    parser.add_argument("--labels", type=Path, default=ROOT / "bench" / "labels.json")
    parser.add_argument("--configs", nargs="+", default=["2:native", "2:448", "1:448", "4:448"], help="fps:width")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "m0_accuracy.jsonl")
    args = parser.parse_args()

    labels = {k: v for k, v in json.loads(args.labels.read_text()).items() if not k.startswith("_")}
    clef = Clef()
    rows = []
    with args.out.open("a") as log:
        for name, label in sorted(labels.items()):
            path = args.clips / name
            if not path.exists():
                continue
            camera = name.rsplit("_", 1)[0].replace("_", " ").strip()
            for config in args.configs:
                fps_text, width = config.split(":")
                fps = float(fps_text)
                clip = path.read_bytes() if width == "native" else prepare_clip(path.read_bytes(), fps, int(width))
                result = clef.ask(f"Security camera '{camera}'. The clip was recorded because motion was detected.",
                                  QUESTIONS, videos=[clip], video_fps=fps)
                person = result.answers["person"]["true"]
                cause = max(result.answers["cause"], key=result.answers["cause"].get)
                row = {"time": time.time(), "clip": name, "config": config, "tokens": result.input_tokens,
                       "seconds": round(result.seconds, 2), "person": round(person, 3), "cause": cause,
                       "cause_p": round(result.answers["cause"][cause], 3), "label": label}
                rows.append(row)
                log.write(json.dumps(row) + "\n")
                mark = "" if label["person"] is None else ("ok " if (person >= 0.5) == label["person"] else "MISS")
                print(f"{name:<44} {config:<9} person {person:.2f} {mark:<4} cause {cause} ({row['cause_p']:.2f})"
                      f"  [{label['nuisance'] or ('person' if label['person'] else '-')}]  {result.seconds:.1f}s")

    print("\nconfig     person acc  mean p(person) yes / no   nuisance top-1   tokens  seconds")
    for config in args.configs:
        sel = [r for r in rows if r["config"] == config]
        known = [r for r in sel if r["label"]["person"] is not None]
        acc = sum((r["person"] >= 0.5) == r["label"]["person"] for r in known) / max(1, len(known))
        yes = [r["person"] for r in known if r["label"]["person"]]
        no = [r["person"] for r in known if not r["label"]["person"]]
        nuis = [r for r in sel if r["label"]["nuisance"]]
        top1 = sum(r["cause"] == r["label"]["nuisance"] for r in nuis) / max(1, len(nuis))
        print(f"{config:<10} {acc:>6.2f} ({len(known)})   {statistics.mean(yes) if yes else 0:.2f} / "
              f"{statistics.mean(no) if no else 0:.2f}        {top1:.2f} ({len(nuis)})      "
              f"{statistics.median(r['tokens'] for r in sel):>6.0f}  {statistics.median(r['seconds'] for r in sel):6.2f}")


if __name__ == "__main__":
    main()
