"""Can EmbeddingGemma 2 score PTZ views the way Clef does, for a fraction of the GPU time?

    cd ~/nx_clef_operator && HF_HOME=~/embed/hf ~/embed/.venv/bin/python -m bench.embed_hunter \
        --grid /tmp/ptzgrid --clef-run /tmp/clefprec/fp8.json

Scores the 40 hand-labelled Back PTZ grid views (bench/clef_precision.py GRID_LABELS: 1 = open yard where people
would be, 0 = wall, roof or close foliage) zero-shot: each view's image embedding against descriptions of good
places and of blockers, softmax over all of them (like CLIP), score = the good share. Reports Spearman agreement with
the labels next to Clef's (from a clef_precision run on the same images), how well it flags the hopeless views, the
time per view and per 9-tile batch, and whether it picks out the picnic-table tiles of p190/t-10.
"""

from __future__ import annotations

import argparse
import io
import json
import statistics
import time
from pathlib import Path

import torch
from PIL import Image

from bench.clef_precision import GRID_LABELS, GRID_PANS, spearman
from clef_operator.hunter import crop_jpeg, prior_from, tile_boxes

GOOD = ["an open lawn or yard where a person could walk", "a path or walkway", "a patio or seating area",
        "a play area or play equipment", "a picnic table", "a person standing outside in a yard"]
BAD = ["a close-up of a house wall or siding", "the roof or eaves of a building", "a pergola or lattice seen from above",
       "tree branches and leaves close to the camera", "the sky", "a blurry close-up of a surface"]


def load(model_id: str, budget: int = 70):
    """Text and image encoders only, bf16, ``budget`` soft tokens per image (70: 9 tiles in 0.35 s next to a busy
    Clef, against 1.3 s at the default 280, and the views ranked as well or better)."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_id, device="cuda", model_kwargs={"torch_dtype": torch.bfloat16},
                                config_kwargs={"audio_config": None})
    processor = model[0].processor
    processor.image_processor.max_soft_tokens = processor.image_processor.image_seq_length = budget
    processor.image_seq_length = budget
    return model


def view_scores(images: torch.Tensor, texts: torch.Tensor, n_good: int, temperature: float) -> list[float]:
    probs = torch.softmax(images @ texts.T / temperature, dim=1)
    return probs[:, :n_good].sum(dim=1).tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--grid", required=True)
    parser.add_argument("--clef-run", help="a clef_precision run JSON on the same grid, for Clef's agreement")
    parser.add_argument("--model", default="google/embeddinggemma-2")
    parser.add_argument("--temperature", type=float, default=0.02)
    args = parser.parse_args()

    model = load(args.model)
    grid = json.loads((Path(args.grid) / "grid.json").read_text())
    rows = [(r["name"], GRID_LABELS[r["tilt"]][GRID_PANS.index(r["pan"])]) for r in grid]
    rows = [(n, lab) for n, lab in rows if lab is not None]
    images = [Image.open(Path(args.grid) / f"{n}.jpg").convert("RGB") for n, _ in rows]
    model.encode(images[:2])  # warm-up
    torch.cuda.synchronize()
    started = time.time()
    img = model.encode(images, convert_to_tensor=True, normalize_embeddings=True, batch_size=8)
    torch.cuda.synchronize()
    per_view = (time.time() - started) / len(images)
    txt = model.encode(GOOD + BAD, prompt_name="query", convert_to_tensor=True, normalize_embeddings=True)
    labels = [lab for _, lab in rows]
    print(f"{len(rows)} labelled views, {per_view * 1000:.0f} ms per view (batched, Clef sharing the GPU)")
    for temperature in (0.01, 0.02, 0.05):
        scores = view_scores(img, txt, len(GOOD), temperature)
        print(f"  temperature {temperature}: embedding rho {spearman(scores, labels):.3f}")
    scores = view_scores(img, txt, len(GOOD), args.temperature)
    if args.clef_run:
        run = {r["id"]: r for r in json.loads(Path(args.clef_run).read_text())["rows"] if r["repeat"] == 0}
        clef = [prior_from(run[n]["answers"])[0] for n, _ in rows]
        print(f"  Clef scene score rho {spearman(clef, labels):.3f} (same views)")
        both = [0.5 * a + 0.5 * b for a, b in zip(scores, clef)]
        print(f"  mean of the two rho {spearman(both, labels):.3f}")
    print("\nview          label  embed")
    for (name, lab), s in sorted(zip(rows, scores), key=lambda x: x[1]):
        print(f"  {name:<14} {lab:4.1f}  {s:.2f}")
    walls = [s for (_, lab), s in zip(rows, scores) if lab == 0]
    yards = [s for (_, lab), s in zip(rows, scores) if lab >= 0.7]
    cut = max(walls) if walls else 0.0
    print(f"\nhopeless (label 0): embed {min(walls):.2f}-{max(walls):.2f}; open yard (label >= 0.7): "
          f"{min(yards):.2f}-{max(yards):.2f}; yard views below the worst wall's score: "
          f"{sum(s < cut for s in yards)}/{len(yards)}")
    walls_sorted = sorted(walls)
    for keep in (1.0, 0.9, 0.75, 0.5):
        threshold = walls_sorted[int(len(walls_sorted) * keep) - 1]
        lost = sum(s <= threshold for s in yards)
        print(f"  skipping views at or below {threshold:.2f} drops {int(len(walls_sorted) * keep)}/{len(walls)} walls "
              f"and {lost}/{len(yards)} yard views")

    jpeg = (Path(args.grid) / "p190.0_t-10.jpg").read_bytes()
    tiles = [Image.open(io.BytesIO(crop_jpeg(jpeg, *box))).convert("RGB") for box in tile_boxes(3)]
    torch.cuda.synchronize()
    started = time.time()
    tile_emb = model.encode(tiles, convert_to_tensor=True, normalize_embeddings=True, batch_size=9)
    torch.cuda.synchronize()
    print(f"\n9 tiles of p190/t-10 in {(time.time() - started) * 1000:.0f} ms")
    cues = model.encode(["a picnic table", "a path or walkway", "a person"], prompt_name="query", convert_to_tensor=True,
                        normalize_embeddings=True)
    sims = (tile_emb @ cues.T).tolist()
    for i, (box, row) in enumerate(zip(tile_boxes(3), sims)):
        print(f"  tile {i} at {box[0]:.2f},{box[1]:.2f}: picnic {row[0]:.3f}  path {row[1]:.3f}  person {row[2]:.3f}")
    print(f"  median picnic {statistics.median(r[0] for r in sims):.3f}")


if __name__ == "__main__":
    main()
