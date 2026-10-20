"""EmbeddingGemma 2 for the PTZ hunter's cheap looks (``scripts/embed_server.py``, next to Clef on the DGX).

Clef answers questions; an embedding only says how close an image is to a description. That is enough for two of the
hunter's jobs, where a mistake costs search time and never a missed person (Clef still asks the hunt questions on
every look):

- Scoring a new view: a view much closer to "a wall", "the roof", "the sky", "leaves close to the camera" than to "an
  open lawn", "a path", "a patio" is hopeless, and Clef's 20 scene questions are not asked about it.
- Finding where to zoom in: the 9 tiles of a promising view against the helper cues, in one batch (0.35 s), instead
  of a Clef request per tile (about 7 s); a tile that is itself hopeless does not become a zoomed view.

Measured on the 37 hand-labelled Back PTZ views (bench/embed_hunter.py): the good-place share ranks them with
Spearman 0.75 (Clef's scene score: 0.84); walls, roofs and close foliage scored 0.01-0.25 and open yard 0.38-0.82, two
half-open views 0.12. ``HOPELESS`` (0.10) drops about two thirds of the walls and none of the open or half-open
views. As a person detector it is not good enough (clip frames: AUC 0.90 overall, 0.53 on Front porch), so it never
stands in for Clef's hunt questions.
"""

from __future__ import annotations

import base64
import json
import math
import time
import urllib.request

GOOD = ["an open lawn or yard where a person could walk", "a path or walkway", "a patio or seating area",
        "a play area or play equipment", "a picnic table", "a person standing outside in a yard"]
BAD = ["a close-up of a house wall or siding", "the roof or eaves of a building", "a pergola or lattice seen from above",
       "tree branches and leaves close to the camera", "the sky", "a blurry close-up of a surface"]
HOPELESS = 0.10  # good-place share below which a view is hopeless (walls 0.01-0.25; half-open views from 0.12)
TEMPERATURE = 0.02
TILE_STANDOUT = 0.04  # a tile's similarity to a helper cue this far above the view's median tile stands out
DOWN_FOR_S = 60.0  # after a failed call, do not try again for this long (the hunter falls back to Clef)


# What a view or tile mostly shows, so the hunter never looks at what it should not: the classes below, from these
# descriptions and the camera's own labelled examples (``ContentModel``). Measured on 249 hand-labelled Back PTZ frames
# and tiles, day and infrared night (bench/content_eval.py, leave-one-out): 129 of 140 trees, bushes, sky, siding and
# pergola/roof views caught, 8 of 109 open views wrongly flagged (85% exact class). The good-place share before it caught
# 69 (and only 3 of 23 pergola or roof views, 4 of 11 bushes) at its hopeless threshold.
CONTENT = {
    "tree": ["tree branches and leaves filling the view", "the canopy of a tree", "a close-up of an evergreen tree",
             "leaves glowing white in an infrared night image", "a tree trunk close up"],
    "bush": ["a bush or shrub seen from above", "a hedge or bushes close to the camera", "shrubs and flowers in a garden bed"],
    "sky": ["the sky", "clouds in the sky", "a dark night sky", "a bright overexposed white sky"],
    "siding": ["the siding or wall of a house close up", "a window or door of a house", "a close-up of a wooden fence post"],
    "roof": ["a pergola or lattice seen from above", "the roof or eaves of a house", "wooden slats seen from above"],
    "grass": ["a lawn of grass", "grass and fallen leaves on the ground", "bare ground at night", "a mowed lawn seen from above"],
    "yard": ["an open backyard with a fence", "a playground or picnic area", "a park with a lawn and trees in the distance",
             "a path through a yard", "a picnic table on a concrete slab"],
}
UNWANTED = frozenset({"tree", "bush", "sky", "siding", "roof", "stay away"})  # "stay away": the operator's descriptions
CONTENT_DIMS = 256  # vectors cut to this (EmbeddingGemma is trained to allow it): as accurate, a third of the work
UNWANTED_HOPELESS = 0.75  # a view this covered by things not to look at is never hunted
UNWANTED_TILE = 0.5  # a tile this likely to be one is never zoomed in on
# Not worth a close-up either: plain lawn. A tight zoom on grass shows nothing unless someone is there, and then
# motion, Clef or the camera's own detection brings the hunter in anyway.
DULL = UNWANTED | {"grass"}
DULL_CLOSEUP = 0.5  # a close-up (or a tile it would come from) this likely to be dull is not taken


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def good_share(image: list[float], good: list[list[float]], bad: list[list[float]],
               temperature: float = TEMPERATURE) -> float:
    """0..1: how much more the image looks like the good places than the blockers (softmax over all of them)."""
    logits = [dot(image, t) / temperature for t in good + bad]
    top = max(logits)
    weights = [math.exp(x - top) for x in logits]
    return sum(weights[:len(good)]) / sum(weights)


def shorten(vector: list[float], dims: int = CONTENT_DIMS) -> list[float]:
    head = vector[:dims]
    norm = math.sqrt(sum(x * x for x in head)) or 1.0
    return [x / norm for x in head]


def _softmax(scores: dict[str, float]) -> dict[str, float]:
    top = max(scores.values())
    weights = {k: math.exp(v - top) for k, v in scores.items()}
    total = sum(weights.values())
    return {k: w / total for k, w in weights.items()}


class ContentModel:
    """What an image mostly shows: half from the class descriptions (zero-shot), half from the camera's labelled
    examples (its ``k`` nearest), when it has any."""

    def __init__(self, prompts: dict[str, list[list[float]]], examples: list[tuple[list[float], str]] | None = None,
                 k: int = 7, temperature: float = TEMPERATURE) -> None:
        self.prompts = {c: [shorten(v) for v in vs] for c, vs in prompts.items() if vs}
        self.examples = [(shorten(v), c) for v, c in examples or [] if c in self.prompts]
        self.k, self.temperature = k, temperature

    def classify(self, vector: list[float]) -> dict[str, float]:
        v = shorten(vector)
        text = _softmax({c: max(dot(v, p) for p in ps) / self.temperature for c, ps in self.prompts.items()})
        if not self.examples:
            return text
        nearest = sorted(((dot(v, e), c) for e, c in self.examples), reverse=True)[:self.k]
        votes = {c: 0.0 for c in self.prompts}
        for similarity, c in nearest:
            votes[c] += math.exp(similarity / self.temperature)
        total = sum(votes.values()) or 1.0
        return {c: 0.5 * text[c] + 0.5 * votes[c] / total for c in self.prompts}

    def unwanted(self, vector: list[float]) -> float:
        """The chance it mostly shows something not to look at."""
        return sum(p for c, p in self.classify(vector).items() if c in UNWANTED)

    def dull(self, vector: list[float]) -> float:
        """The chance it mostly shows something not worth a close-up (that, or plain lawn)."""
        return sum(p for c, p in self.classify(vector).items() if c in DULL)


def standout_tiles(sims: list[dict[str, float]]) -> list[tuple[int, float, str]]:
    """(tile index, score 0..1, cue) for tiles whose similarity to a helper cue stands out from the view's median tile
    by ``TILE_STANDOUT`` or more, best first (the score reaches 1 at 2.5 times that)."""
    if not sims:
        return []
    medians = {c: sorted(t[c] for t in sims)[len(sims) // 2] for c in sims[0]}
    found = []
    for i, tile in enumerate(sims):
        best = max(((s - medians[c], c) for c, s in tile.items()), default=None)
        if best and best[0] >= TILE_STANDOUT:
            found.append((i, min(1.0, best[0] / (2.5 * TILE_STANDOUT)), best[1]))
    return sorted(found, key=lambda f: -f[1])


class Embedder:
    """Client of the embedding server. Text embeddings are cached (the descriptions and cues rarely change). Any call
    may raise; after a failure the embedder is down for ``DOWN_FOR_S`` and raises at once (no waiting on timeouts)."""

    def __init__(self, url: str, timeout: float = 15.0) -> None:
        self.url, self.timeout = url.rstrip("/"), timeout
        self._texts: dict[tuple[str, str], list[float]] = {}
        self._down_until = 0.0
        self.calls = 0

    def _post(self, body: dict) -> dict:
        if time.time() < self._down_until:
            raise ConnectionError("embedding server down")
        request = urllib.request.Request(f"{self.url}/embed", json.dumps(body).encode(),
                                         {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                self.calls += 1
                return json.load(response)
        except Exception:
            self._down_until = time.time() + DOWN_FOR_S
            raise

    def images(self, jpegs: list[bytes]) -> list[list[float]]:
        return self._post({"images": [base64.b64encode(j).decode() for j in jpegs]})["vectors"]

    def look(self, jpeg: bytes, boxes: list[tuple[float, float, float, float]],
             thumb: int = 320) -> tuple[list[float], list[list[float]], bytes]:
        """One look in one call: the frame's vector, each tile's (cut by the server), and a thumbnail JPEG."""
        reply = self._post({"images": [base64.b64encode(jpeg).decode()], "crops": [list(b) for b in boxes],
                            "thumb": thumb})
        vectors = reply["vectors"]
        return vectors[0], vectors[1:], base64.b64decode(reply.get("thumb", ""))

    def texts(self, texts: list[str], prompt: str = "query") -> list[list[float]]:
        missing = [t for t in texts if (prompt, t) not in self._texts]
        if missing:
            for text, vector in zip(missing, self._post({"texts": missing, "prompt": prompt})["vectors"]):
                self._texts[(prompt, text)] = vector
        return [self._texts[(prompt, t)] for t in texts]

    def content_model(self, examples: list[tuple[list[float], str]] | None = None,
                      stay_away: list[str] | None = None) -> ContentModel:
        """The content classes (plus the operator's stay-away descriptions as one more unwanted class), with a
        camera's labelled examples."""
        prompts = {c: self.texts(texts) for c, texts in CONTENT.items()}
        if stay_away:
            prompts["stay away"] = self.texts(list(stay_away))
        return ContentModel(prompts, examples)

    def share(self, vector: list[float]) -> float:
        """The good-place share of an image's vector."""
        return good_share(vector, self.texts(GOOD), self.texts(BAD))

    def likeness(self, frame: list[float], tiles: list[list[float]], wanted: list[str],
                 avoid: list[str]) -> tuple[float, dict[str, float], dict[str, float]]:
        """One look against the hunter's descriptions: the frame's good-place share; for each wanted description
        (what to look for, where to look closer) the best similarity over the frame and its tiles (small things show
        in a tile); for each stay-away description the frame's similarity (a whole view of it, not a corner)."""
        share = self.share(frame)
        like = {d: max(dot(v, t) for v in [frame] + tiles) for d, t in zip(wanted, self.texts(wanted))} if wanted else {}
        unlike = {d: dot(frame, t) for d, t in zip(avoid, self.texts(avoid))} if avoid else {}
        return share, like, unlike

    def tile_scores(self, vectors: list[list[float]], cues: list[str]) -> tuple[list[float], list[dict[str, float]]]:
        """For each tile's vector: its good-place share, and its similarity to each helper cue."""
        good, bad, cue_vectors = self.texts(GOOD), self.texts(BAD), self.texts(cues)
        shares = [good_share(v, good, bad) for v in vectors]
        sims = [{c: dot(v, cv) for c, cv in zip(cues, cue_vectors)} for v in vectors]
        return shares, sims
