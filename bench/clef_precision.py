"""Profile the Clef server before and after a precision change (e.g. bf16 -> FP8): speed, memory and answers.

    python3 -m bench.clef_precision prepare --clips /tmp/m0clips --grid /tmp/ptzgrid --cache /tmp/clefprec
    python3 -m bench.clef_precision run --cache /tmp/clefprec --out /tmp/clefprec/bf16.json
    python3 -m bench.clef_precision compare /tmp/clefprec/bf16.json /tmp/clefprec/fp8.json

``prepare`` encodes every input once (so the run itself needs Clef to itself for only a few minutes): the M0
hand-labelled clips (bench/labels.json) at 4 and 1 fps, the 40 hand-labelled Back PTZ grid views, and a 9-tile
image request. ``run`` sends them all, recording seconds, tokens and every answer, plus Clef's GPU memory.
``compare`` reports speed, memory, person accuracy, scene-ranking agreement with the labels, and how far answers moved.
Run it with the gateway stopped, or other traffic shares Clef's lock and inflates the times.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
from pathlib import Path

from bench.m0_accuracy import QUESTIONS as CLIP_QUESTIONS
from clef_operator.clef import Clef, image_data_url, noul, video_data_url
from clef_operator.env import ROOT
from clef_operator.gateway import MAX_PIXELS, prepare_clip, prepare_image
from clef_operator.hunter import crop_jpeg, likelihood_questions, prior_from, tile_boxes

CLIP_CONFIGS = [(4.0, 448), (1.0, 448)]
# Hand labels for the Back PTZ grid views (1 = open yard where people would be, 0 = wall, roof or close foliage).
GRID_PANS = [72, 101.5, 131, 160.5, 190, 219.5, 249, 278.5]
GRID_LABELS = {0: [0, 0, .5, 1, 1, .5, 0, 0], -10: [0, 0, .3, 1, 1, None, None, None],
               -20: [0, 0, 0, .5, .7, .5, .7, .3], -30: [0, 0, 0, 0, .2, .3, .7, .2], -40: [0, 0, 0, 0, 0, .3, .6, .2]}
SCENE_STATE = "A snapshot from the PTZ security camera 'Back PTZ'. PTZ over a back yard: lawn, trees, a pergola."
TILE_STATE = "Snapshots from a PTZ security camera over a back yard. The images are 9 overlapping tiles of one snapshot."


def prepare(args: argparse.Namespace) -> None:
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    labels = {k: v for k, v in json.loads((ROOT / "bench" / "labels.json").read_text()).items() if not k.startswith("_")}
    items = []
    for name, label in sorted(labels.items()):
        path = Path(args.clips) / name
        if not path.exists():
            continue
        for fps, width in CLIP_CONFIGS:
            out = cache / f"{path.stem}_{fps:g}fps.mp4"
            out.write_bytes(prepare_clip(path.read_bytes(), fps, width))
            items.append({"kind": "clip", "id": f"{name}@{fps:g}", "file": str(out), "fps": fps, "label": label,
                          "camera": name.rsplit("_", 1)[0].replace("_", " ").strip()})
    grid = json.loads((Path(args.grid) / "grid.json").read_text())
    for row in grid:
        label = GRID_LABELS[row["tilt"]][GRID_PANS.index(row["pan"])]
        out = cache / f"{row['name']}.jpg"
        out.write_bytes(prepare_image((Path(args.grid) / f"{row['name']}.jpg").read_bytes(), MAX_PIXELS))
        items.append({"kind": "scene", "id": row["name"], "file": str(out), "label": label})
    jpeg = (Path(args.grid) / "p190.0_t-10.jpg").read_bytes()
    tiles = []
    for i, box in enumerate(tile_boxes(3)):
        out = cache / f"tile{i}.jpg"
        out.write_bytes(prepare_image(crop_jpeg(jpeg, *box), MAX_PIXELS))
        tiles.append(str(out))
    items.append({"kind": "tiles", "id": "tiles@190/-10", "files": tiles})
    (cache / "items.json").write_text(json.dumps(items, indent=1))
    print(f"prepared {len(items)} items in {cache}")


def clef_gpu_mib() -> float | None:
    """GPU memory the Clef server process holds (nvidia-smi; on GB10 the per-process figure is the one reported)."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    sizes = [float(line.split(",")[-1]) for line in out.splitlines() if line.strip()]
    return max(sizes) if sizes else None


def run(args: argparse.Namespace) -> None:
    clef = Clef(args.clef)
    items = json.loads((Path(args.cache) / "items.json").read_text())
    scene_questions = likelihood_questions(["Is there a person in view?"])
    rows = []
    for repeat in range(args.repeats):
        for item in items:
            if item["kind"] == "clip":
                result = clef.ask(f"Security camera '{item['camera']}'. The clip was recorded because motion was detected.",
                                  CLIP_QUESTIONS, videos=[video_data_url(Path(item["file"]).read_bytes())],
                                  video_fps=item["fps"])
            elif item["kind"] == "scene":
                result = clef.ask(SCENE_STATE, scene_questions, images=[Path(item["file"]).read_bytes()])
            else:
                result = clef.ask(TILE_STATE, {"t": noul("Is there a picnic table in any of these images?")},
                                  images=[Path(f).read_bytes() for f in item["files"]])
            rows.append({"id": item["id"], "kind": item["kind"], "repeat": repeat, "seconds": round(result.seconds, 3),
                         "tokens": result.input_tokens, "answers": result.answers, "label": item.get("label")})
            print(f"{item['id']:<44} {result.seconds:6.2f}s {result.input_tokens:6d} tok", flush=True)
    health = clef.health()
    Path(args.out).write_text(json.dumps({"time": time.time(), "url": args.clef, "gpu_mib": clef_gpu_mib(),
                                          "health": health, "rows": rows}, indent=1))
    print(f"wrote {args.out}")


def spearman(xs: list[float], ys: list[float]) -> float:
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0] * len(v)
        for k, i in enumerate(order):
            r[i] = k
        return r
    a, b = rank(xs), rank(ys)
    n, ma, mb = len(a), sum(a) / len(a), sum(b) / len(b)
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b)) ** 0.5


def summarize(path: str) -> dict:
    data = json.loads(Path(path).read_text())
    rows = [r for r in data["rows"] if r["repeat"] == 0]
    out = {"gpu_mib": data.get("gpu_mib")}
    for kind, label in (("clip", None), ("scene", None), ("tiles", None)):
        sel = [r for r in data["rows"] if r["kind"] == kind]
        out[f"{kind}_seconds"] = statistics.median(r["seconds"] for r in sel) if sel else None
    for fps in ("4", "1"):
        sel = [r for r in data["rows"] if r["kind"] == "clip" and r["id"].endswith(f"@{fps}")]
        out[f"clip{fps}_seconds"] = statistics.median(r["seconds"] for r in sel) if sel else None
    clips = [r for r in rows if r["kind"] == "clip" and r["label"]["person"] is not None]
    out["person_acc"] = sum((r["answers"]["person"]["true"] >= 0.5) == r["label"]["person"] for r in clips) / len(clips)
    yes = [r["answers"]["person"]["true"] for r in clips if r["label"]["person"]]
    no = [r["answers"]["person"]["true"] for r in clips if not r["label"]["person"]]
    out["person_p_yes"], out["person_p_no"] = statistics.mean(yes), statistics.mean(no)
    nuis = [r for r in rows if r["kind"] == "clip" and r["label"]["nuisance"]]
    out["nuisance_top1"] = sum(max(r["answers"]["cause"], key=r["answers"]["cause"].get) == r["label"]["nuisance"]
                               for r in nuis) / max(1, len(nuis))
    scenes = [r for r in rows if r["kind"] == "scene" and r["label"] is not None]
    out["scene_rho"] = spearman([prior_from(r["answers"])[0] for r in scenes], [r["label"] for r in scenes])
    out["rows"] = {r["id"]: r for r in rows}
    return out


def compare(args: argparse.Namespace) -> None:
    a, b = summarize(args.before), summarize(args.after)
    print(f"{'':28s} {'before':>10s} {'after':>10s}")
    for key, unit in (("gpu_mib", " MiB"), ("clip4_seconds", " s"), ("clip1_seconds", " s"), ("scene_seconds", " s"),
                      ("tiles_seconds", " s"), ("person_acc", ""), ("person_p_yes", ""), ("person_p_no", ""),
                      ("nuisance_top1", ""), ("scene_rho", "")):
        va, vb = a.get(key), b.get(key)
        fa = "-" if va is None else f"{va:.3f}{unit}"
        fb = "-" if vb is None else f"{vb:.3f}{unit}"
        change = f"  ({(vb / va - 1) * 100:+.0f}%)" if va and vb and unit else ""
        print(f"{key:28s} {fa:>10s} {fb:>10s}{change}")
    deltas, flips = [], []
    for rid, ra in a["rows"].items():
        rb = b["rows"].get(rid)
        if not rb:
            continue
        for q, ans in ra["answers"].items():
            if q in rb["answers"]:
                for option, p in ans.items():
                    if option == "false":
                        continue
                    deltas.append(abs(p - rb["answers"][q].get(option, 0.0)))
                if "true" in ans and (ans["true"] >= 0.5) != (rb["answers"][q]["true"] >= 0.5):
                    flips.append(f"{rid}:{q} {ans['true']:.2f}->{rb['answers'][q]['true']:.2f}")
    deltas.sort()
    print(f"\nanswer probability change over {len(deltas)} answers: median {statistics.median(deltas):.4f}, "
          f"95th {deltas[int(len(deltas) * 0.95)]:.4f}, max {deltas[-1]:.4f}")
    print(f"yes/no answers that flipped at 0.5: {len(flips)}" + ("".join(f"\n   {f}" for f in flips[:15])))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--clips", required=True)
    p.add_argument("--grid", required=True)
    p.add_argument("--cache", required=True)
    r = sub.add_parser("run")
    r.add_argument("--cache", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--clef", default="http://127.0.0.1:8100")
    r.add_argument("--repeats", type=int, default=2)
    c = sub.add_parser("compare")
    c.add_argument("before")
    c.add_argument("after")
    args = parser.parse_args()
    {"prepare": prepare, "run": run, "compare": compare}[args.command](args)


if __name__ == "__main__":
    main()
