"""What a PTZ camera has seen, by direction: an EmbeddingGemma vector for every spot the hunter has looked at, in the
camera's own polar space (pan, tilt, field of view), searchable with plain text.

Each look adds its frame at the view and its 9 tiles at the views they cover (centred on the tile, at the tile's
share of the field of view), so the map is finer than the views the hunter visits. A later look at the same spot
replaces the earlier one. Each entry points into a small thumbnail of its look (the frame, with the tile's box), so a
search can show what matched. ``search`` ranks entries by cosine similarity to a text query's vector.

Saved as JSON in the gateway's state directory (vectors as base64 float16, 1.5 KB each); thumbnails as JPEG files
next to it, removed once no entry points at them.
"""

from __future__ import annotations

import base64
import json
import struct
import threading
import time
from pathlib import Path

from .hunter import View, aim_at, same_spot

SAVE_EVERY_S = 300.0


def pack(vector: list[float]) -> str:
    return base64.b64encode(struct.pack(f"<{len(vector)}e", *vector)).decode()


def unpack(text: str) -> list[float]:
    data = base64.b64decode(text)
    return list(struct.unpack(f"<{len(data) // 2}e", data))


def tile_view(view: View, box: tuple[float, float, float, float]) -> View:
    """The direction a tile of ``view`` shows: centred on it, at its share of the field of view."""
    centred = aim_at(view, box, 0.0)
    return View.of(centred.pan, centred.tilt, view.fov * box[2])


class EmbedMap:
    def __init__(self, path: Path | None = None, limit: int = 4000) -> None:
        self.path, self.limit = path, limit
        self.thumbs = path.with_suffix("") if path else None  # a folder of look thumbnails beside the JSON
        self.entries: list[dict] = []
        self._lock = threading.Lock()
        self._saved_at = time.time()
        if path and path.exists():
            data = json.loads(path.read_text())
            self.entries = [{**e, "vec": unpack(e["vec"])} for e in data.get("entries", [])]

    def add_look(self, view: View, frame: list[float], tiles: list[list[float]],
                 boxes: list[tuple[float, float, float, float]], thumb: bytes, now: float | None = None) -> None:
        now = time.time() if now is None else now
        shot = f"{int(now * 1000)}"
        if self.thumbs and thumb:
            self.thumbs.mkdir(parents=True, exist_ok=True)
            (self.thumbs / f"{shot}.jpg").write_bytes(thumb)
        added = [{"pan": view.pan, "tilt": view.tilt, "fov": view.fov, "t": now, "vec": frame, "shot": shot,
                  "box": [0.0, 0.0, 1.0, 1.0], "kind": "view"}]
        for box, vector in zip(boxes, tiles):
            at = tile_view(view, box)
            added.append({"pan": at.pan, "tilt": at.tilt, "fov": at.fov, "t": now, "vec": vector, "shot": shot,
                          "box": list(box), "kind": "tile"})
        with self._lock:
            for entry in added:
                spot = View(entry["pan"], entry["tilt"], entry["fov"])
                self.entries = [e for e in self.entries if not same_spot(View(e["pan"], e["tilt"], e["fov"]), spot)]
                self.entries.append(entry)
            if len(self.entries) > self.limit:
                self.entries = sorted(self.entries, key=lambda e: e["t"])[-self.limit:]
        if now - self._saved_at >= SAVE_EVERY_S:
            self.save()

    def search(self, queries: list[list[float]], top: int = 8, negatives: list[list[float]] | None = None) -> dict:
        """Every entry's similarity to the queries (the best of them: "a person; a vehicle" finds either) for the map,
        and the ``top`` matches, best first. With ``negatives``, the score is how much more an entry looks like the
        best query than like the closest negative: with nobody in view, the directions closest to "a person" are
        bushes; "a person or a path, not a bush" ranks the places worth watching."""
        def sim(vectors: list[list[float]], e: dict) -> float:
            return max(sum(a * b for a, b in zip(q, e["vec"])) for q in vectors)
        with self._lock:
            scored = [(sim(queries, e) - (sim(negatives, e) if negatives else 0.0), e) for e in self.entries]
        scored.sort(key=lambda se: -se[0])
        public = lambda s, e: {"pan": round(e["pan"], 1), "tilt": round(e["tilt"], 1), "fov": round(e["fov"], 1),  # noqa: E731
                               "score": round(s, 4), "t": e["t"], "shot": e["shot"], "box": e["box"], "kind": e["kind"]}
        best: list[dict] = []
        for s, e in scored:  # the top matches, one per spot (a tile and the view it is in are nearly the same place)
            if len(best) >= top:
                break
            if all(not same_spot(View(e["pan"], e["tilt"], max(e["fov"], b["fov"])),
                                 View(b["pan"], b["tilt"], max(e["fov"], b["fov"])), share=0.5) for b in best):
                best.append(public(s, e))
        return {"entries": len(scored), "map": [[round(e["pan"], 1), round(e["tilt"], 1), round(e["fov"], 1),
                                                  round(s, 4)] for s, e in scored], "top": best}

    def thumb(self, shot: str) -> bytes | None:
        if not self.thumbs or not shot.isdigit():
            return None
        path = self.thumbs / f"{shot}.jpg"
        return path.read_bytes() if path.exists() else None

    def save(self) -> None:
        self._saved_at = time.time()
        if not self.path:
            return
        with self._lock:
            entries = [{**e, "vec": pack(e["vec"])} for e in self.entries]
            used = {e["shot"] for e in self.entries}
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"version": 1, "entries": entries}))
        tmp.replace(self.path)
        if self.thumbs and self.thumbs.exists():
            for old in self.thumbs.glob("*.jpg"):
                if old.stem not in used:
                    old.unlink(missing_ok=True)
