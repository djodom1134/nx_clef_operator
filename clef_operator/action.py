"""Motion while a PTZ holds still: where a view always moves (boring), and action worth zooming in on.

The plugin streams the Server's motion grid (44 x 32 cells, OR-ed over half a second) to the gateway for cameras with
a hunter. It means something only while the camera is still: during a move every cell moves.

Per view, each cell's share of frames with motion is learned (an average over about ``LEARN_FRAMES`` frames, kept
across visits): a cell that moves in ``BORING`` of frames or more is waving trees, flapping flags, IR noise at night,
and is ignored. Action is a cluster of other cells that moves for ``PERSIST`` frames running (a second: one noisy frame
is not action) and either is new there (nothing near it a few seconds before) or moves fast across the grid. Until a
view has been watched ``MIN_SEEN`` frames, nothing counts: every tree would look new.
"""

from __future__ import annotations

import base64
from collections import deque

from .motion import clusters

GRID_W, GRID_H = 44, 32
BORING = 0.12  # a cell moving in this share of frames is boring
LEARN_FRAMES = 600  # the share is an average over about this many frames (five minutes at 2 a second)
MIN_SEEN = 60  # frames watched at a view before anything there counts as action (30 s)
MIN_CELLS = 2  # cells in a cluster: one cell alone is usually noise
PERSIST = 2  # frames running a cluster must show in
NEAR_CELLS = 4.0  # a cluster this close to one in the previous frame is the same thing moving
ONSET_S = (2.0, 6.0)  # "new": nothing near it this many seconds before
FAST_CELLS_S = 1.5  # or moving at least this many cells a second
NUISANCE_BUSY = 0.06  # a direction moving in this share of still frames is a nuisance (gusts move leaves too seldom
# for BORING: on Back PTZ the leaves chased all morning moved in 9-11% of frames; 6% flags 2.5% of the scene)
CHASE_HALF_LIFE_S = 86400.0  # a chase that found nobody counts against its direction for about a day
CHASE_RADIUS = 5.0  # degrees: chases this close are the same nuisance (at least: see Hunter._fruitless)
CHASE_SPAN_M = 1.0  # or this far apart on the ground: close to the camera a bush spans far more degrees
NUISANCE_CHASES = 1.8  # fruitless chases (fading) near a direction before motion there is not chased: two recent


def decode(cells_b64: str) -> set[tuple[int, int]]:
    """The plugin's grid (bit ``r * 44 + c``, least significant bit first) as (column, row) cells."""
    data = base64.b64decode(cells_b64)
    return {(i % GRID_W, i // GRID_W) for i in range(min(len(data) * 8, GRID_W * GRID_H)) if data[i // 8] >> (i % 8) & 1}


def encode(cells: set[tuple[int, int]]) -> str:
    data = bytearray((GRID_W * GRID_H + 7) // 8)
    for c, r in cells:
        i = r * GRID_W + c
        data[i // 8] |= 1 << (i % 8)
    return base64.b64encode(bytes(data)).decode()


def centroid(cluster: set[tuple[int, int]]) -> tuple[float, float]:
    return sum(c for c, _ in cluster) / len(cluster), sum(r for _, r in cluster) / len(cluster)


def box(cluster: set[tuple[int, int]], grow: float = 1.5) -> tuple[float, float, float, float]:
    """Normalized (x, y, w, h) around the cluster, grown by ``grow`` (room for where it is going), inside the frame."""
    cs, rs = [c for c, _ in cluster], [r for _, r in cluster]
    w, h = (max(cs) - min(cs) + 1) / GRID_W * grow, (max(rs) - min(rs) + 1) / GRID_H * grow
    w, h = min(1.0, max(w, 3 / GRID_W)), min(1.0, max(h, 3 / GRID_H))
    cx, cy = (min(cs) + max(cs) + 1) / 2 / GRID_W, (min(rs) + max(rs) + 1) / 2 / GRID_H
    return min(max(0.0, cx - w / 2), 1.0 - w), min(max(0.0, cy - h / 2), 1.0 - h), w, h


class ViewMotion:
    """What one view's motion usually looks like: each cell's share of frames with motion (0-255), and how many
    frames it has been watched."""

    def __init__(self, busy: bytes | None = None, seen: int = 0) -> None:
        self.busy = bytearray(busy) if busy and len(busy) == GRID_W * GRID_H else bytearray(GRID_W * GRID_H)
        self.seen = seen

    def boring(self, cell: tuple[int, int]) -> bool:
        return self.busy[cell[1] * GRID_W + cell[0]] >= BORING * 255

    def add(self, cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
        """Learn from one frame; returns its cells that are not boring (judged before learning from it)."""
        interesting = {cell for cell in cells if not self.boring(cell)}
        n = min(self.seen + 1, LEARN_FRAMES)  # a plain average at first, a moving one after
        for i in range(GRID_W * GRID_H):
            moved = 255 if (i % GRID_W, i // GRID_W) in cells else 0
            self.busy[i] = max(0, min(255, round(self.busy[i] + (moved - self.busy[i]) / n)))
        self.seen += 1
        return interesting

    def to_json(self) -> dict:
        return {"busy": base64.b64encode(bytes(self.busy)).decode(), "seen": self.seen}

    @staticmethod
    def from_json(d: dict | None) -> "ViewMotion":
        if not d:
            return ViewMotion()
        return ViewMotion(base64.b64decode(d.get("busy", "")), int(d.get("seen", 0)))


class NuisanceMap:
    """Where in the camera's world (pan and tilt, 1-degree cells) motion is a nuisance: leaves, bushes, shadows in the
    wind. Shared by every view and kept across restarts, where ``ViewMotion`` is one view's own and needs minutes at
    that view to learn (the hunter holds a view 20-45 s at a time, and the leaves seen from one view looked new from
    the next). Two things are learned:

    - how often each direction moves while the camera is still (a share of frames, as ``ViewMotion``): a direction
      moving in ``NUISANCE_BUSY`` of frames or more is a nuisance from any view that sees it;
    - chases that found nobody, by direction, fading with a ``CHASE_HALF_LIFE_S`` half-life: where they add up to
      ``NUISANCE_CHASES`` (two recent ones) since anything was last found nearby, motion there is not chased again (gusts move leaves
      now and then, too seldom to count as busy, but each gust looks new)."""

    def __init__(self, cells: dict[tuple[int, int], list[int]] | None = None,
                 chases: list[tuple[float, float, float]] | None = None) -> None:
        self.cells = cells or {}  # (pan, tilt) cell -> [busy 0-255, frames seen]
        self.chases = chases or []  # (time, pan, tilt) of chases that found nobody
        self._keys: tuple[tuple, list[tuple[int, int]]] | None = None

    @staticmethod
    def key(pan: float, tilt: float) -> tuple[int, int]:
        return round(pan) % 360, round(tilt)

    def _frame_keys(self, view) -> list[tuple[int, int]]:
        """The world cell under each grid cell of ``view`` (index ``r * 44 + c``), cached for the view."""
        if self._keys is None or self._keys[0] != (view.pan, view.tilt, view.fov):
            keys = [self.key(view.pan + ((c + 0.5) / GRID_W - 0.5) * view.fov,
                             view.tilt + (0.5 - (r + 0.5) / GRID_H) * view.fov * 0.75)
                    for r in range(GRID_H) for c in range(GRID_W)]
            self._keys = ((view.pan, view.tilt, view.fov), keys)
        return self._keys[1]

    def busy(self, view) -> set[tuple[int, int]]:
        """The grid cells of ``view`` that look at a nuisance direction (judged before learning from this frame)."""
        keys = self._frame_keys(view)
        busy = set()
        for i, k in enumerate(keys):
            cell = self.cells.get(k)
            if cell and cell[1] >= MIN_SEEN and cell[0] >= NUISANCE_BUSY * 255:
                busy.add((i % GRID_W, i // GRID_W))
        return busy

    def observe(self, view, cells: set[tuple[int, int]]) -> None:
        """Learn from one still frame at ``view``: each direction in it moved or not (any of its grid cells)."""
        keys = self._frame_keys(view)
        moved = {keys[r * GRID_W + c] for c, r in cells}
        for k in set(keys):
            cell = self.cells.setdefault(k, [0, 0])
            n = min(cell[1] + 1, LEARN_FRAMES)
            cell[0] = max(0, min(255, round(cell[0] + ((255 if k in moved else 0) - cell[0]) / n)))
            cell[1] += 1

    def chased(self, now: float, pan: float, tilt: float) -> None:
        self.chases.append((now, pan, tilt))
        self.chases = [c for c in self.chases if now - c[0] < 7 * CHASE_HALF_LIFE_S][-500:]

    def fruitless(self, now: float, pan: float, tilt: float, found_since: float, radius: float = CHASE_RADIUS) -> float:
        """How many chases (fading) within ``radius`` degrees found nobody, counting only those after the last find
        nearby."""
        return sum(0.5 ** ((now - t) / CHASE_HALF_LIFE_S) for t, p, q in self.chases
                   if t > found_since and abs((p - pan + 180) % 360 - 180) <= radius and abs(q - tilt) <= radius)

    def to_json(self) -> dict:
        return {"cells": [[k[0], k[1], v[0], v[1]] for k, v in self.cells.items()],
                "chases": [list(c) for c in self.chases]}

    @staticmethod
    def from_json(d: dict | None) -> "NuisanceMap":
        if not d:
            return NuisanceMap()
        return NuisanceMap({(int(p), int(t)): [int(b), int(n)] for p, t, b, n in d.get("cells", [])},
                           [tuple(c) for c in d.get("chases", [])])


class ActionWatch:
    """Frames at one view, in order: reports action (a box in the frame) when it starts."""

    def __init__(self, motion: ViewMotion) -> None:
        self.motion = motion
        self.frames: deque[tuple[float, list[set[tuple[int, int]]]]] = deque(maxlen=20)

    def add(self, now: float, cells: set[tuple[int, int]],
            ignore: set[tuple[int, int]] = frozenset()) -> tuple[float, float, float, float] | None:
        """``ignore``: cells known elsewhere to be a nuisance (``NuisanceMap``)."""
        interesting = self.motion.add(cells) - ignore
        found = [c for c in clusters(interesting) if len(c) >= MIN_CELLS]
        self.frames.append((now, found))
        if self.motion.seen < MIN_SEEN or not found:
            return None
        for cluster in sorted(found, key=len, reverse=True):
            if self._is_action(now, cluster):
                return box(cluster)
        return None

    def _near(self, cluster: set[tuple[int, int]], others: list[set[tuple[int, int]]]) -> set[tuple[int, int]] | None:
        cx, cy = centroid(cluster)
        best = None
        for other in others:
            ox, oy = centroid(other)
            d = ((cx - ox) ** 2 + (cy - oy) ** 2) ** 0.5
            if d <= NEAR_CELLS and (best is None or d < best[0]):
                best = (d, other)
        return best[1] if best else None

    def _is_action(self, now: float, cluster: set[tuple[int, int]]) -> bool:
        frames = list(self.frames)
        # Persistent: the same thing in each of the previous PERSIST - 1 frames (within a second and a half).
        track, current = [cluster], cluster
        for t, found in reversed(frames[-PERSIST:-1]):
            if now - t > 1.5:
                return False
            previous = self._near(current, found)
            if previous is None:
                return False
            track.append(previous)
            current = previous
        # New there: nothing near it a few seconds before.
        before = [found for t, found in frames if ONSET_S[0] <= now - t <= ONSET_S[1]]
        new = bool(before) and all(self._near(cluster, found) is None for found in before)
        # Or fast: its centre moved quickly over the persistent frames.
        (x1, y1), (x0, y0) = centroid(track[0]), centroid(track[-1])
        span = max(0.5, now - frames[-len(track)][0])
        fast = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5 / span >= FAST_CELLS_S
        return new or fast
