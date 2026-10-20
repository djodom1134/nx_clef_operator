"""How well does the hunter tell what not to look at (trees, bushes, sky, siding, roof) from open ground?

    python3 bench/content_sheets.py            # (embedding venv) contact sheets of map entries to label by eye
    python3 -m bench.content_eval --index /tmp/label_sheets/index.json \\
        --labels bench/content_labels/back_ptz_map.json --grid /tmp/ptzgrid \\
        --grid-labels bench/content_labels/back_ptz_grid.json [--save ~/.clef_operator/content_labels_<camera>.json]

Labels are by eye, one class per image (its dominant content): tree, bush, sky, siding (a house wall, window, door,
or a fence post close up), roof (roof, eaves, or a pergola or lattice from above), grass, yard (open ground with
something in it: a fence line, path, play or picnic area). Images that were too mixed to call were left out.
Compares the deployed good-place share (A), the class descriptions in ``embed.CONTENT`` (B), the labelled examples
alone, each image judged by its 7 nearest others (C, leave one out), and both (D, what the hunter uses), at
``embed.CONTENT_DIMS``. ``--save`` writes the labelled examples for a camera's hunter (the gateway loads them).

On Back PTZ (249 images, day and infrared night): A caught 69 of 140 unwanted (3 of 23 pergola/roof views), D 129
with 8 of 109 open views wrongly flagged.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import urllib.request
from pathlib import Path

from clef_operator.embed import BAD, CONTENT, CONTENT_DIMS, GOOD, UNWANTED, ContentModel, good_share, shorten
from clef_operator.embed_map import pack, unpack


def embed(url: str, body: dict) -> list[list[float]]:
    request = urllib.request.Request(f"{url}/embed", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.load(response)["vectors"]


def load(args: argparse.Namespace) -> list[tuple[list[float], str, str]]:
    data = []
    labels = json.loads(Path(args.labels).read_text())
    for item in json.loads(Path(args.index).read_text()):
        if str(item["id"]) in labels:
            vec = unpack(item["vec"]) if isinstance(item["vec"], str) else item["vec"]
            data.append((vec, labels[str(item["id"])], f"m{item['id']}"))
    if args.grid and args.grid_labels:
        names = [g["name"] for g in json.loads((Path(args.grid) / "grid.json").read_text())]
        grid_labels = json.loads(Path(args.grid_labels).read_text())
        chosen = [(i, n) for i, n in enumerate(names) if f"g{i}" in grid_labels]
        images = [base64.b64encode((Path(args.grid) / f"{n}.jpg").read_bytes()).decode() for _, n in chosen]
        for (i, _), vec in zip(chosen, embed(args.embed, {"images": images})):
            data.append((vec, grid_labels[f"g{i}"], f"g{i}"))
    return data


def auc(pos: list[float], neg: list[float]) -> float:
    return sum((a > b) + 0.5 * (a == b) for a in pos for b in neg) / (len(pos) * len(neg))


def report(name: str, data, scores: list[float], flagged: list[bool]) -> None:
    pos = [s for s, (_, lab, _) in zip(scores, data) if lab in UNWANTED]
    neg = [s for s, (_, lab, _) in zip(scores, data) if lab not in UNWANTED]
    caught = sum(f and lab in UNWANTED for f, (_, lab, _) in zip(flagged, data))
    wrong = sum(f and lab not in UNWANTED for f, (_, lab, _) in zip(flagged, data))
    per = ", ".join(f"{c} {sum(f for f, (_, lab, _) in zip(flagged, data) if lab == c)}/{sum(lab == c for _, lab, _ in data)}"
                    for c in CONTENT)
    print(f"{name:<38} AUC {auc(pos, neg):.3f} | caught {caught}/{len(pos)} unwanted, wrongly flagged {wrong}/{len(neg)}"
          f"\n{'':40}flagged per class: {per}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--grid")
    parser.add_argument("--grid-labels")
    parser.add_argument("--embed", default="http://127.0.0.1:8300")
    parser.add_argument("--save", help="write the labelled examples here for the camera's hunter")
    args = parser.parse_args()
    data = load(args)
    print(f"{len(data)} labelled images, {sum(lab in UNWANTED for _, lab, _ in data)} unwanted, {CONTENT_DIMS} dims")
    good = embed(args.embed, {"texts": GOOD, "prompt": "query"})
    bad = embed(args.embed, {"texts": BAD, "prompt": "query"})
    share = [good_share(v, good, bad) for v, _, _ in data]
    report("A deployed good-place share < 0.10", data, [1 - s for s in share], [s < 0.10 for s in share])
    prompts = {c: embed(args.embed, {"texts": texts, "prompt": "query"}) for c, texts in CONTENT.items()}
    text_only = ContentModel(prompts)
    b = [text_only.unwanted(v) for v, _, _ in data]
    report("B class descriptions", data, b, [x >= 0.5 for x in b])
    d, exact = [], 0
    for i, (vec, label, _) in enumerate(data):
        model = ContentModel(prompts, [(v, lab) for j, (v, lab, _) in enumerate(data) if j != i])
        probs = model.classify(vec)
        exact += max(probs, key=probs.get) == label
        d.append(sum(p for c, p in probs.items() if c in UNWANTED))
    report("D descriptions + examples (leave one out)", data, d, [x >= 0.5 for x in d])
    print(f"D exact class {exact / len(data):.2f}")
    # Close-ups: plain lawn is dull too (nothing to see close up); yard with something in it is not.
    dull = [m for m in (ContentModel(prompts, [(v, lab) for j, (v, lab, _) in enumerate(data) if j != i]).dull(vec)
                        for i, (vec, _, _) in enumerate(data))]
    for c in ("grass", "yard"):
        flagged = sum(x >= 0.5 for x, (_, lab, _) in zip(dull, data) if lab == c)
        print(f"close-up check: {c} judged not worth a close-up {flagged}/{sum(lab == c for _, lab, _ in data)}")
    if args.save:
        rows = [{"vec": pack(shorten(v)), "label": lab, "source": name} for v, lab, name in data]
        Path(args.save).expanduser().write_text(json.dumps({"version": 1, "dims": CONTENT_DIMS, "examples": rows}))
        print(f"saved {len(rows)} examples to {args.save}")


if __name__ == "__main__":
    main()
