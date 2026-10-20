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


class ActionWatch:
    """Frames at one view, in order: reports action (a box in the frame) when it starts."""

    def __init__(self, motion: ViewMotion) -> None:
        self.motion = motion
        self.frames: deque[tuple[float, list[set[tuple[int, int]]]]] = deque(maxlen=20)

    def add(self, now: float, cells: set[tuple[int, int]]) -> tuple[float, float, float, float] | None:
        interesting = self.motion.add(cells)
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
