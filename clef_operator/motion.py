"""Nx motion grid: mask parsing and writing, per-window motion summaries, and motion rank (PRD: Priority and scheduling).

Nx stores a camera's motion mask as rectangles on a 44 x 32 grid: ``"sensitivity,x,y,w,h;..."``. Sensitivity 0
masks a cell (no motion detected there); 1..9 are detection sensitivities. Cells no rectangle covers use the default.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

GRID_W, GRID_H = 44, 32
DEFAULT_SENSITIVITY = 5

Grid = list[list[int]]  # [y][x]


def empty_grid(value: int = 0) -> Grid:
    return [[value] * GRID_W for _ in range(GRID_H)]


def parse_mask(mask: str, default: int = DEFAULT_SENSITIVITY) -> Grid:
    grid = empty_grid(default)
    for rect in filter(None, (part.strip() for part in mask.split(";"))):
        sensitivity, x, y, w, h = (int(v) for v in rect.split(","))
        for row in range(max(0, y), min(GRID_H, y + h)):
            for col in range(max(0, x), min(GRID_W, x + w)):
                grid[row][col] = sensitivity
    return grid


def format_mask(grid: Grid) -> str:
    """Encode a sensitivity grid as Nx rectangles: runs along each row, merged down while rows repeat."""
    rects: list[tuple[int, int, int, int, int]] = []
    open_runs: dict[tuple[int, int, int], int] = {}  # (x, w, sensitivity) -> index in rects, for runs ending on row y-1
    for y in range(GRID_H):
        row, runs, x = grid[y], [], 0
        while x < GRID_W:
            start, value = x, row[x]
            while x < GRID_W and row[x] == value:
                x += 1
            runs.append((start, x - start, value))
        next_open = {}
        for start, width, value in runs:
            key = (start, width, value)
            if key in open_runs:
                index = open_runs[key]
                s, rx, ry, rw, rh = rects[index]
                rects[index] = (s, rx, ry, rw, rh + 1)
            else:
                index = len(rects)
                rects.append((value, start, y, width, 1))
            next_open[key] = index
        open_runs = next_open
    return ";".join(",".join(str(v) for v in rect) for rect in rects)


class MotionRank(IntEnum):
    """Lower runs first within a priority class."""

    ONSET = 0
    ACTIVE = 1
    STILL = 2


@dataclass
class WindowMotion:
    """Motion inside one window, from the Server's per-frame motion grids."""

    frames: int = 0
    frames_with_motion: int = 0
    heat: Grid = field(default_factory=empty_grid)  # frames with motion, per cell
    onset: bool = False  # motion started inside the window after a quiet period
    frame_cells: list[set[tuple[int, int]]] = field(default_factory=list)  # cells that moved, per frame with motion
    frame_index: list[int] = field(default_factory=list)  # which frame each entry of frame_cells came from

    def add_frame(self, cells: set[tuple[int, int]]) -> None:
        self.frames += 1
        if cells:
            self.frames_with_motion += 1
            self.frame_cells.append(set(cells))
            self.frame_index.append(self.frames - 1)
            for x, y in cells:
                self.heat[y][x] += 1

    def share(self, ignore: Grid | None = None, only: Grid | None = None) -> float:
        """Share of frames with any motion in cells in ``only`` and not in ``ignore``. Counted per frame, not per
        cell, so a car crossing a zone quickly counts even though each cell it passes moves for only a few frames."""
        if not self.frames:
            return 0.0
        if ignore is None and only is None:
            return self.frames_with_motion / self.frames
        moving = sum(
            1 for cells in self.frame_cells
            if any(not (ignore and ignore[y][x]) and not (only and not only[y][x]) for x, y in cells)
        )
        return moving / self.frames

    def longest_run(self, ignore: Grid | None = None, only: Grid | None = None) -> int:
        """Most consecutive frames with motion in the allowed cells."""
        longest = run = 0
        previous = -2
        for index, cells in zip(self.frame_index, self.frame_cells):
            if not any(not (ignore and ignore[y][x]) and not (only and not only[y][x]) for x, y in cells):
                continue
            run = run + 1 if index == previous + 1 else 1
            previous = index
            longest = max(longest, run)
        return longest

    def rank(self, active_share: float = 0.05, ignore: Grid | None = None, only: Grid | None = None,
             fps: float = 7.0) -> MotionRank:
        """Active with ``active_share`` of frames moving, or half a second of continuous motion (a passing car is
        continuous but small; noise is scattered single frames)."""
        if self.share(ignore, only) < active_share and self.longest_run(ignore, only) < max(2, 0.5 * fps):
            return MotionRank.STILL
        return MotionRank.ONSET if self.onset else MotionRank.ACTIVE

    def cells(self, min_frames: int = 1) -> set[tuple[int, int]]:
        return {(x, y) for y in range(GRID_H) for x in range(GRID_W) if self.heat[y][x] >= min_frames}


def bounding_box(cells: set[tuple[int, int]]) -> tuple[float, float, float, float] | None:
    """Normalized (x, y, w, h) of the cells, for cropping a clip to where the motion is."""
    if not cells:
        return None
    xs, ys = [x for x, _ in cells], [y for _, y in cells]
    x0, y0, x1, y1 = min(xs), min(ys), max(xs) + 1, max(ys) + 1
    return x0 / GRID_W, y0 / GRID_H, (x1 - x0) / GRID_W, (y1 - y0) / GRID_H


def clusters(cells: set[tuple[int, int]]) -> list[set[tuple[int, int]]]:
    """Group cells into 8-connected clusters."""
    remaining, out = set(cells), []
    while remaining:
        stack, cluster = [remaining.pop()], set()
        while stack:
            x, y = stack.pop()
            cluster.add((x, y))
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    neighbour = (x + dx, y + dy)
                    if neighbour in remaining:
                        remaining.remove(neighbour)
                        stack.append(neighbour)
        out.append(cluster)
    return out
