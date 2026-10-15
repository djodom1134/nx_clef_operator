"""PTZ hunter (PRD: PTZ hunter): moves a PTZ camera, slowly and quietly, to views where a question is likely to be
answered "yes", zooms in on what it finds, and dwells there.

Everything is in the camera's logical space as the Nx Server reports it: pan and tilt in degrees, and zoom as the
horizontal field of view in degrees. (On Back PTZ the normalized tilt is not even monotonic with the real tilt, so
moves judged in normalized units went the wrong way.)

Survey: a serpentine sweep across the pan range at the widest zoom, in rows a little above, at and a little below
home's tilt (within the tilt limits), plus short rows near low magnets. Each view is scored with Clef image questions:
the hunt questions, then how much open ground a person could stand on is in view, and blockers (a nearby wall, a roof
or pergola below, sky, close foliage). Each snapshot is also cut into a 3 x 3 grid of full-resolution tiles and the
hunt questions are asked of all tiles in one request; only if that says maybe are they asked tile by tile, to find where.
Small, distant subjects are not lost when the frame is shrunk for Clef, and an empty view costs two requests, not ten.

Hunt: choose the next view by likelihood, past hits at this hour, staleness, exploration and magnet pull, minus a
travel penalty. Magnets are views operators care about (home, presets, listed views); the pull grows the longer the
hunter is away from all of them.

Found: when the frame or a tile answers yes (or a tile comes close), centre on that tile and zoom in, slowly, then
dwell while the answer stays yes, and raise an Nx generic event.

Moves are slow: continuous mode at a capped speed, axes timed to arrive together, a full stop on arrival, with a rest
between moves; slower passes close the gap, and a small absolute move (a short hop: real cameras ignore the speed of
absolute moves) corrects the last few degrees. An axis that stops moving while driven is at its end stop (on Back PTZ,
tilting up into the end stop makes the camera pan by itself). Views are recorded where the camera actually ended up.
"""

from __future__ import annotations

import json
import math
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Protocol

from .clef import choice, noul

Answers = dict[str, dict[str, float]]
AskImage = Callable[[str, "bytes | list[bytes]", dict[str, dict]], Answers]  # (camera_id, jpeg(s), questions) -> answers
Crop = Callable[[bytes, float, float, float, float], bytes]  # (jpeg, x, y, w, h normalized) -> jpeg

# Degrees per second at continuous speed 1.0, measured on Back PTZ (Milesight MS-C5341-X42PE) at speed 0.2: pan 6,
# tilt 3.5, field of view about 9.5 degrees a second. Used to time the axes so they arrive together and to stop
# ahead of the target; positions are always read back from the camera.
PAN_RATE, TILT_RATE, FOV_RATE = 30.0, 17.5, 48.0
PAN_TOL, TILT_TOL, FOV_TOL = 0.6, 0.6, 1.0
READBACK_LAG_S = 0.5  # the Server's position is refreshed about once a second: stop this much travel early
STALL_S = 2.5  # a driven axis whose position has not changed for this long is at its end stop
MEMORY_VERSION = 2  # views scored by older prompts (or at positions never reached) are forgotten on load


def angle_delta(a: float, b: float) -> float:
    """Signed shortest pan difference b - a, in degrees."""
    return (b - a + 540.0) % 360.0 - 180.0


@dataclass(frozen=True)
class View:
    pan: float  # degrees
    tilt: float  # degrees, up is positive
    fov: float  # horizontal field of view, degrees (smaller = zoomed in)

    @staticmethod
    def of(pan: float, tilt: float, fov: float) -> "View":
        return View(round(pan % 360.0 * 2) / 2, round(tilt * 2) / 2, round(fov * 2) / 2)

    def distance(self, other: "View") -> float:
        return abs(angle_delta(self.pan, other.pan)) + abs(self.tilt - other.tilt) + 0.5 * abs(self.fov - other.fov)


@dataclass
class HunterConfig:
    camera_id: str
    camera_name: str = ""
    description: str = ""
    interests: str = ""
    targets: list[str] = field(default_factory=lambda: ["Is there a person in view?"])
    threshold: float = 0.7
    curious: float = 0.4  # a tile at least this sure gets a closer look
    dwell_s: float = 60.0
    focus_s: float = 4.0  # after a zoom change, no commands or snapshots for this long: the camera refocuses
    wide_share: float = 0.75  # share of the time spent zoomed all the way out; the rest, looking closer at spots
    hold_s: float = 52.0  # while the camera's own windows are arriving, stay this long (one full 30 s window, its
    #                       20 s step and the settle) unless a clean window was answered sooner
    max_dwell_s: float = 600.0  # then move on even if the answer is still yes
    settle_s: float = 2.0
    max_speed: float = 0.1  # continuous-move speed cap, 0..1 (about 1 degree a second at 0.1 on Back PTZ)
    rest_s: float = 20.0  # minimum pause between moves
    pan_min: float = 0.0  # degrees; a span of 360 means the full circle
    pan_max: float = 360.0
    tilt_min: float = -90.0  # degrees; targets are kept inside, clear of the end stops
    tilt_max: float = 90.0
    tilts: list[float] = field(default_factory=list)  # survey tilts in degrees; empty = from home and magnets
    overlap: float = 0.8  # survey step as a share of the field of view
    zoom_in: bool = True
    min_fov: float = 6.0
    tiles: int = 3  # tile grid per side for small-subject search; 1 = whole frame only
    tile_gate: float = 0.15  # all tiles go to Clef in one request first; each is asked alone only if this sure
    magnets: list[dict] = field(default_factory=list)  # [{"pan", "tilt", "fov"?}] in degrees
    presets: list[str] = field(default_factory=list)  # preset ids used as magnets (positions learned once)
    home_preset: str = ""  # the camera's normal view; empty = the first home ever seen (remembered), not where it is now
    magnet_weight: float = 0.5
    magnet_radius: float = 25.0  # degrees
    magnet_return_s: float = 600.0
    avoid: list[str] = field(default_factory=list)  # extra negative prompts
    cues: list[str] = field(default_factory=lambda: list(DEFAULT_CUES))  # places where a yes is more likely nearby


# Learning where the target shows up. Each look (a snapshot, or one of the camera's own windows while the hunter holds
# the view) adds its P(target). A view's yield starts from its scene score, as if PRIOR_LOOKS looks had each shown
# the target with probability PRIOR_RATE x score, so evidence takes over after a few dozen looks. A view looked at at
# least QUIET_LOOKS times whose yield is under QUIET_SHARE of the best view's is quiet: skipped, except for a recheck
# every QUIET_RECHECK_S in case that changes (a gate left open, a new path).
PRIOR_RATE = 0.1
PRIOR_LOOKS = 8.0
HOUR_WEIGHTS = {0: 1.0, 1: 0.5, 2: 0.25}
OTHER_HOURS_WEIGHT = 0.05
NOISE_P = 0.05  # Clef's P(target) on empty scenes is 0.005-0.05: below this a look counts as nothing seen
QUIET_LOOKS = 12
QUIET_SHARE = 0.25  # and only once the target has been seen somewhere (SIGHTINGS): "rare compared with where it shows
SIGHTINGS = 1.0  # up", not tiny differences in Clef's noise while nothing shows up anywhere
QUIET_RECHECK_S = 3 * 3600.0


@dataclass
class ViewStats:
    prior: float = 0.5
    avoid: float = 0.0
    visits: int = 0
    hits: dict[int, int] = field(default_factory=dict)  # hour -> yes count
    checks: dict[int, int] = field(default_factory=dict)  # hour -> checks
    last_visit: float = 0.0
    last_yes: float = 0.0
    scored: bool = False  # prior and avoid come from Clef (not the default, e.g. after Clef dropped the question)
    looks: dict[int, int] = field(default_factory=dict)  # hour -> looks (snapshots and the camera's own windows)
    seen: dict[int, float] = field(default_factory=dict)  # hour -> summed P(target) over those looks
    scene: float = -1.0  # the scene score before helper answers (-1: not scored yet)
    cues: dict[str, float] = field(default_factory=dict)  # helper question -> P(yes) for the whole view
    cue_scan_at: float = 0.0  # when this view's tiles were last scanned for helper answers
    found_by: str = ""  # for a zoomed view: the helper question its tile stood out for
    base_avoid: float = 0.0  # the built-in blockers alone (``avoid`` adds the operator's, weighed per camera)
    extra_avoid: dict[str, float] = field(default_factory=dict)  # operator's avoid question -> P(yes)

    def hit_rate(self, hour: int) -> float:
        return (self.hits.get(hour, 0) + 0.5) / (self.checks.get(hour, 0) + 2.0)

    def observe(self, hour: int, p: float) -> None:
        self.looks[hour] = self.looks.get(hour, 0) + 1
        self.seen[hour] = self.seen.get(hour, 0.0) + max(0.0, p - NOISE_P) / (1.0 - NOISE_P)

    @property
    def total_looks(self) -> int:
        return sum(self.looks.values())

    def yield_rate(self, hour: int) -> float:
        """How likely a look here shows the target at this hour: the scene score, worth ``PRIOR_LOOKS`` looks, blended
        with what looks here actually showed, nearby hours counting less."""
        looks = seen = 0.0
        for h, n in self.looks.items():
            w = HOUR_WEIGHTS.get(min((h - hour) % 24, (hour - h) % 24), OTHER_HOURS_WEIGHT)
            looks += w * n
            seen += w * self.seen.get(h, 0.0)
        return (PRIOR_LOOKS * PRIOR_RATE * self.prior + seen) / (PRIOR_LOOKS + looks)

    @property
    def total_hits(self) -> int:
        return sum(self.hits.values())

    def to_json(self) -> dict:
        return {"prior": self.prior, "avoid": self.avoid, "visits": self.visits, "hits": self.hits,
                "checks": self.checks, "last_visit": self.last_visit, "last_yes": self.last_yes, "scored": self.scored,
                "looks": self.looks, "seen": self.seen, "scene": self.scene, "cues": self.cues,
                "cue_scan_at": self.cue_scan_at, "found_by": self.found_by, "base_avoid": self.base_avoid,
                "extra_avoid": self.extra_avoid}

    @staticmethod
    def from_json(d: dict) -> "ViewStats":
        if "base_avoid" not in d:
            # Saved before the operator's avoid questions were weighed per camera: its avoid may hold a veto from one
            # that is yes everywhere ("Is there a tree in view?"). Score it again on its next visit.
            d = {**d, "scored": False, "avoid": 0.0, "base_avoid": 0.0}
        return ViewStats(d["prior"], d["avoid"], d["visits"], {int(k): v for k, v in d["hits"].items()},
                         {int(k): v for k, v in d["checks"].items()}, d["last_visit"], d["last_yes"], d.get("scored", False),
                         {int(k): v for k, v in d.get("looks", {}).items()}, {int(k): v for k, v in d.get("seen", {}).items()},
                         d.get("scene", d["prior"] if d.get("scored") else -1.0), d.get("cues", {}),
                         d.get("cue_scan_at", 0.0), d.get("found_by", ""), d.get("base_avoid", d["avoid"]),
                         d.get("extra_avoid", {}))


# View scoring, checked against 40 hand-labelled Back PTZ views (rank correlation +0.75; the earlier prompts scored
# -0.49 and ranked the house wall first). "Is most of this view vegetation?" answers yes for a lawn, so the open
# ground a person would stand on is asked about directly, and foliage only costs half: trees framing a lawn are fine.
GROUND = "Does this view show open ground where a person could walk or stand, such as a lawn, path, patio, yard, driveway or street?"
FOLIAGE = "Is most of this view blocked by trees, bushes or leaves close to the camera, so that little open ground can be seen?"
DEFAULT_AVOID = [  # hard blockers: a view that is mostly one of these is skipped
    "Is most of this view taken up by a building's wall, windows, doors or roof close to the camera, so that little open ground can be seen?",
    "Is the camera looking down at a roof, pergola, lattice or structure close below it that hides most of the ground?",
    "Is most of this view sky?",
]
# Helper questions: places where the hunt questions are more often answered yes. Only an answer that stands out
# counts: "Is there a yard?" says yes to almost every view of a back yard, "Is there a gate?" to one tile.
DEFAULT_CUES = [
    "Is there a path or walkway?",
    "Is there a gate?",
    "Is there a play area or play equipment?",
    "Is there a picnic table, patio or seating area?",
    "Is there a yard?",
]
CUE_STANDOUT_TILE = 0.25  # a tile counts when a helper answer is this far above the view's median tile, and
CUE_MIN_TILE = 0.5  # at least this sure
CUE_RESCAN_S = 6 * 3600.0  # a wide view's tiles are scanned for helper answers at most this often
ZOOM_CHILDREN = 3  # zoomed views added per scan, at most
VIEW_KINDS = {  # what fills the view; the last three are blockers too (catches a wall the yes/no question misses)
    "ground": "open ground such as a lawn, yard, path, patio or street",
    "plants": "trees, bushes or leaves close to the camera",
    "building": "a building's wall, windows, doors or roof close to the camera",
    "below": "a roof, pergola or lattice right below the camera",
    "sky": "sky",
}


def likelihood_questions(targets: list[str], interests: str = "", avoid: list[str] | None = None,
                         cues: list[str] | None = None) -> dict[str, dict]:
    """Where could the targets be seen (open ground, helper places), and what blocks the view (views to avoid)."""
    questions = {"_ground": noul(GROUND), "_foliage": noul(FOLIAGE),
                 "_kind": choice("What fills most of this view?", VIEW_KINDS)}
    for i, text in enumerate(cues or []):
        questions[f"_cue{i}"] = noul(text)
    if interests:
        questions["_interest"] = noul(f"Would this view help watch for the following: {interests}")
    for i, text in enumerate(DEFAULT_AVOID + list(avoid or [])):
        questions[f"_avoid{i}"] = noul(text)
    return questions


def prior_from(answers: Answers) -> tuple[float, float]:
    """(likelihood, avoid): open ground, scaled down by the strongest built-in blocker and by half the foliage. The
    operator's extra avoid questions (``_avoid`` beyond the built-in ones) are weighed per camera, in ``Hunter``."""
    p = {q: a["true"] for q, a in answers.items() if q.startswith("_") and "true" in a}
    kind = answers.get("_kind", {})
    avoid = max([v for q, v in p.items() if q.startswith("_avoid") and int(q[6:]) < len(DEFAULT_AVOID)]
                + [kind.get(k, 0.0) for k in ("building", "below", "sky")] + [0.0])
    likelihood = p.get("_ground", 0.5) * (1.0 - avoid) * (1.0 - 0.5 * p.get("_foliage", 0.0))
    if "_interest" in p:
        likelihood *= 0.75 + 0.25 * p["_interest"]
    return likelihood, avoid


def camera_medians(answers: list[dict[str, float]], min_views: int = 4) -> dict[str, float]:
    """Per question, the median answer over a camera's views (none until ``min_views`` views have answers)."""
    if len(answers) < min_views:
        return {}
    out = {}
    for question in {q for a in answers for q in a}:
        values = sorted(a[question] for a in answers if question in a)
        out[question] = values[len(values) // 2]
    return out


def cue_boost(cues: dict[str, float], medians: dict[str, float]) -> float:
    """0.5..1: how much a view's helper answers stand out from the camera's usual answers (0.5 when none do)."""
    standout = max([max(0.0, p - medians.get(c, 0.0)) for c, p in cues.items()] + [0.0])
    return 0.5 + 0.5 * min(1.0, 2.0 * standout)


def standout_tiles(tiles: list[dict[str, float]]) -> list[tuple[int, float, str]]:
    """(tile index, score, helper question) for tiles where a helper answer stands out from the view's other tiles,
    best first."""
    if not tiles:
        return []
    cues = tiles[0].keys()
    medians = {c: sorted(t[c] for t in tiles)[len(tiles) // 2] for c in cues}
    found = []
    for i, tile in enumerate(tiles):
        best = max(((p, c) for c, p in tile.items() if p >= CUE_MIN_TILE and p - medians[c] >= CUE_STANDOUT_TILE),
                   default=None)
        if best:
            found.append((i, best[0], best[1]))
    return sorted(found, key=lambda f: -f[1])


def zoom_on(view: View, box: tuple[float, float, float, float], min_fov: float, factor: float = 0.5) -> View:
    """The view centred on a tile, zoomed in to ``factor`` of the field of view."""
    centred = aim_at(view, box, min_fov)
    return View.of(centred.pan, centred.tilt, max(min_fov, view.fov * factor))


def survey_pans(config: HunterConfig, fov: float) -> list[float]:
    step = max(5.0, fov * config.overlap)
    span = (config.pan_max - config.pan_min) % 360.0 or 360.0
    count = max(1, int(span // step) + (0 if span >= 359.0 else 1))
    return [(config.pan_min + i * step) % 360.0 for i in range(count)]


def clamp_tilt(config: HunterConfig, tilt: float, margin: float = 2.0) -> float:
    """Keep a target tilt clear of the end stops (driving into one can make the camera pan by itself)."""
    return min(config.tilt_max - margin, max(config.tilt_min + margin, tilt))


def survey_tilts(config: HunterConfig, home: View) -> list[float]:
    """Rows swept across the whole pan range, highest first: a little above home's tilt, at it, and a little below,
    inside the tilt limits."""
    if config.tilts:
        return sorted({clamp_tilt(config, t, 0.0) for t in config.tilts}, reverse=True)
    vertical = home.fov * 0.75  # 4:3 frame
    tilts = {clamp_tilt(config, round((home.tilt + d * 0.6 * vertical) / 5) * 5) for d in (1, 0, -1)}
    return sorted((t for t in tilts if -70.0 <= t <= 20.0), reverse=True)


def survey_views(config: HunterConfig, home: View, magnets: list[View] | None = None) -> list[View]:
    """A serpentine sweep at the widest zoom, highest row first, so the camera never makes a long jump. A magnet whose
    tilt is off those rows (a preset aimed low, say) gets its own short row either side of it, not a full sweep of the
    ground. The magnet itself is already a view: a near copy of it would be visited in turn with it, over and over."""
    pans = survey_pans(config, home.fov)
    rows = survey_tilts(config, home)
    views = []
    for i, tilt in enumerate(rows):
        for pan in (pans if i % 2 == 0 else list(reversed(pans))):
            views.append(View.of(pan, tilt, home.fov))
    step = home.fov * config.overlap
    for m in magnets or []:
        tilt = clamp_tilt(config, round(m.tilt / 5) * 5)
        if all(abs(tilt - r) > 2.5 for r in rows):
            views += [View.of(m.pan + d, tilt, home.fov) for d in (-step, step)]
    return views


def magnet_pull(view: View, magnets: list[View], radius: float) -> float:
    if not magnets:
        return 0.0
    return max(math.exp(-(view.distance(m) / radius) ** 2) for m in magnets)


def sightings(stats: dict) -> float:
    return sum(sum(s.seen.values()) for s in stats.values())


def is_quiet(s: ViewStats, hour: int, best_yield: float, seen_anywhere: float = SIGHTINGS) -> bool:
    return (seen_anywhere >= SIGHTINGS and s.total_looks >= QUIET_LOOKS
            and s.yield_rate(hour) < QUIET_SHARE * best_yield)


def resting(s: ViewStats, now: float, hour: int, best_yield: float, seen_anywhere: float) -> bool:
    """A quiet view is skipped until its recheck is due."""
    return is_quiet(s, hour, best_yield, seen_anywhere) and now - s.last_visit < QUIET_RECHECK_S


def choose_next(stats: dict[View, ViewStats], current: View | None, now: float, hour: int,
                travel_weight: float = 0.002, exploration: float = 0.15, magnets: list[View] | None = None,
                magnet_weight: float = 0.5, magnet_radius: float = 25.0, since_magnet_s: float = 0.0,
                magnet_return_s: float = 600.0, hot_weight: float = 0.8, hot_revisit_s: float = 300.0) -> View:
    """Most promising view: learned yield (relative to the best view) and hits, skipping quiet views, plus staleness, exploration, magnet pull and hot-spot pull, minus
    travel (per degree). The magnet pull grows the longer the hunter has been away from every magnet. Hot spots are
    views where the target was found: the more finds, the stronger the pull, and it comes back within minutes of a
    visit (``hot_revisit_s``) instead of the half hour ordinary views take to go stale. Travel costs little (0.2 for
    100 degrees, about 15 s of moving): at 0.01 a degree the best views across the yard were never worth the trip, and
    the hunter bounced between two neighbouring views for hours."""
    most_hits = max([s.total_hits for s in stats.values()] + [1])
    total_visits = sum(s.visits for s in stats.values()) + 1
    pull_strength = magnet_weight * (0.1 + min(1.0, since_magnet_s / magnet_return_s)) if magnets else 0.0
    best_yield = max([s.yield_rate(hour) for s in stats.values() if s.avoid < 0.8] + [1e-9])
    seen_anywhere = sightings(stats)

    def score(view: View) -> float:
        s = stats[view]
        if s.avoid >= 0.8:
            return -1.0
        value = s.yield_rate(hour) / best_yield  # relative to the best view now, so it stays 0..1 as yields fall
        if resting(s, now, hour, best_yield, seen_anywhere):
            return -0.5
        stale = min(1.0, (now - s.last_visit) / 1800.0) if s.last_visit else 1.0
        # Curiosity in proportion to promise: an unvisited wall is not worth a look just for being unvisited.
        explore = exploration * value * math.sqrt(math.log(total_visits) / (s.visits + 1))
        travel = travel_weight * (view.distance(current) if current else 0.0)
        repeat = 0.5 if current and view.distance(current) <= 3.0 else 0.0  # near copies count as staying put
        pull = pull_strength * magnet_pull(view, magnets or [], magnet_radius)
        hot = 0.0
        if s.total_hits:
            fame = math.log1p(s.total_hits) / math.log1p(most_hits)
            due = min(1.0, (now - s.last_visit) / hot_revisit_s) if s.last_visit else 1.0
            hot = hot_weight * fame * due
        return value * (0.5 + 0.5 * stale) + explore + pull + hot - travel - repeat

    return max(stats, key=score)


def tile_boxes(n: int) -> list[tuple[float, float, float, float]]:
    """n x n tiles, each 1.5 tiles wide so subjects on a seam are whole in one of them."""
    if n <= 1:
        return []
    size = min(1.0, 1.5 / n)
    starts = [i * (1.0 - size) / (n - 1) for i in range(n)]
    return [(x, y, size, size) for y in starts for x in starts]


def aim_at(view: View, box: tuple[float, float, float, float], min_fov: float, aspect: float = 0.75) -> View:
    """The view centred on ``box`` (normalized image coords), zoomed so the box fills about two thirds of the frame."""
    x, y, w, h = box
    cx, cy = x + w / 2, y + h / 2
    pan = view.pan + (cx - 0.5) * view.fov
    tilt = view.tilt + (0.5 - cy) * view.fov * aspect
    fov = max(min_fov, min(view.fov, view.fov * max(w, h) * 1.5))
    return View.of(pan, tilt, fov)


def crop_jpeg(jpeg: bytes, x: float, y: float, w: float, h: float) -> bytes:
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = Path(tmp) / "in.jpg", Path(tmp) / "out.jpg"
        src.write_bytes(jpeg)
        subprocess.run(
            ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(src),
             "-vf", f"crop=iw*{w:.4f}:ih*{h:.4f}:iw*{x:.4f}:ih*{y:.4f}", "-q:v", "3", "-y", str(dst)],
            check=True, timeout=30,
        )
        return dst.read_bytes()


class Ptz(Protocol):
    def position(self) -> View: ...  # logical: degrees and field of view
    def continuous(self, pan: float, tilt: float, zoom: float) -> None: ...
    def absolute(self, view: View) -> None: ...  # a fast, exact move: only used for the last few degrees
    def snapshot(self) -> bytes: ...


class NxPtz:
    def __init__(self, nx, camera_id: str) -> None:
        self.nx, self.camera_id = nx, camera_id

    def position(self) -> View:
        p = self.nx._request("GET", f"/rest/v4/devices/{self.camera_id}/ptz/position?type=logical")
        return View(p["pan"], p["tilt"], p["zoom"])

    def continuous(self, pan: float, tilt: float, zoom: float) -> None:
        self.nx.ptz_continuous(self.camera_id, pan, tilt, zoom)

    def absolute(self, view: View) -> None:
        self.nx._request("POST", f"/rest/v4/devices/{self.camera_id}/ptz/position",
                         {"type": "logical", "pan": view.pan, "tilt": view.tilt, "zoom": view.fov, "speed": 0.1})

    def snapshot(self) -> bytes:
        return self.nx.snapshot(self.camera_id)


TOL = {"pan": PAN_TOL, "tilt": TILT_TOL, "fov": FOV_TOL}
RATE = {"pan": PAN_RATE, "tilt": TILT_RATE, "fov": FOV_RATE}


def _left(now: View, target: View) -> dict[str, float]:
    return {"pan": angle_delta(now.pan, target.pan), "tilt": target.tilt - now.tilt, "fov": target.fov - now.fov}


def _pass(ptz: Ptz, target: View, max_speed: float, sleep: Callable[[float], None], timeout_s: float) -> set[str]:
    """One drive toward ``target``, axes timed to arrive together, ended by a full stop as soon as any axis is within
    tolerance (allowing for readback lag), passes its target, moves away from it, stalls at an end stop, or the pass
    runs well over time. Never changes one axis while others move: Back PTZ sometimes ignores a command that sets one
    axis to zero and keeps panning at full speed; only the full stop is always obeyed. Returns the axes that stalled."""
    remaining = _left(ptz.position(), target)
    moving = {axis: abs(d) > max(TOL[axis], 2.0) for axis, d in remaining.items()}  # nearly there: left to the finish
    if not any(moving.values()):
        return set()
    duration = max(abs(remaining[a]) / (RATE[a] * max_speed) for a in remaining if moving[a])
    speeds, lead = {}, {}
    for axis, d in remaining.items():
        speed = min(max_speed, abs(d) / (RATE[axis] * duration)) if moving[axis] else 0.0
        sign = math.copysign(1.0, d)
        speeds[axis] = (-sign if axis == "fov" else sign) * speed  # zoom in (positive) narrows the field of view
        lead[axis] = TOL[axis] + speed * RATE[axis] * READBACK_LAG_S
    direction = {axis: math.copysign(1.0, d) for axis, d in remaining.items()}
    driven = [axis for axis in speeds if speeds[axis]]
    stalled: set[str] = set()
    last, unchanged = dict(remaining), {axis: 0.0 for axis in remaining}
    ptz.continuous(speeds["pan"], speeds["tilt"], speeds["fov"])
    elapsed, limit = 0.0, min(timeout_s, 1.5 * duration + 5.0)
    try:
        while elapsed < limit:
            sleep(0.25)
            elapsed += 0.25
            left = _left(ptz.position(), target)
            for axis in driven:
                unchanged[axis] = unchanged[axis] + 0.25 if abs(left[axis] - last[axis]) < 0.01 else 0.0
                last[axis] = left[axis]
                if unchanged[axis] >= STALL_S:
                    stalled.add(axis)
            if stalled or any(abs(left[a]) <= lead[a] or math.copysign(1.0, left[a]) != direction[a]
                              or abs(left[a]) > abs(remaining[a]) + 2.0 for a in driven):
                break
    finally:
        ptz.continuous(0.0, 0.0, 0.0)
    return stalled


def move_slowly(ptz: Ptz, target: View, max_speed: float, sleep: Callable[[float], None] = time.sleep,
                timeout_s: float = 300.0, passes: int = 3, finish_within: float = 10.0) -> View:
    """Drive toward ``target`` at no more than ``max_speed`` and stop there. Each continuous pass stops a little early
    (the Server reports positions late); once within ``finish_within`` degrees on every axis (or after the last pass,
    so a view is never learned where the camera did not get to), an absolute move puts the camera exactly there (a
    short hop: absolute moves ignore their speed). An axis that stalled at an end stop
    is left where it is. Returns where the camera really is."""
    stuck: set[str] = set()
    goal = target
    for i in range(passes + 1):
        if i:
            sleep(1.0)  # let the readback catch up before judging the last pass
        now = ptz.position()
        left = _left(now, target)
        goal = View(now.pan if "pan" in stuck else target.pan, now.tilt if "tilt" in stuck else target.tilt,
                    now.fov if "fov" in stuck else target.fov)
        if all(abs(left[a]) <= TOL[a] or a in stuck for a in left):
            return now
        if i == passes or all(abs(left[a]) <= finish_within or a in stuck for a in left):
            break
        stuck |= _pass(ptz, goal, max_speed, sleep, timeout_s)
    try:
        ptz.absolute(goal)
    except Exception:  # noqa: BLE001 - a camera without absolute moves stays where the passes left it
        return ptz.position()
    sleep(2.0)
    return ptz.position()


@dataclass
class HunterStatus:
    state: str = "starting"  # waiting for PTZ | surveying | hunting | zooming | dwelling | holding | returning | stopped | error
    view: View | None = None
    last_answers: dict[str, float] = field(default_factory=dict)
    found: str = ""
    error: str = ""
    moves: int = 0


class Hunter:
    """One per PTZ camera; runs in its own thread until stopped."""

    def __init__(self, config: HunterConfig, ptz: Ptz, ask: AskImage,
                 on_found: Callable[[HunterConfig, View, str, float], None] | None = None,
                 locate_preset: Callable[[str], View | None] | None = None, crop: Crop = crop_jpeg,
                 sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.time,
                 memory: Path | None = None) -> None:
        self.config, self.ptz, self.ask, self.on_found = config, ptz, ask, on_found
        self.locate_preset, self.crop, self.memory = locate_preset, crop, memory
        self.sleep, self.clock = sleep, clock
        self.stats: dict[View, ViewStats] = {}
        self.status = HunterStatus()
        self.home: View | None = None
        self.magnets: list[View] = []
        rejected = []
        for m in config.magnets:
            try:
                pan, tilt = float(m["pan"]), float(m["tilt"])
            except (KeyError, TypeError, ValueError):
                rejected.append(str(m))
                continue
            if -90.0 <= tilt <= 90.0:
                self.magnets.append(View.of(pan, tilt, float(m.get("fov", 0.0) or 0.0)))
            else:
                rejected.append(f"{pan:g},{tilt:g}")
        self.warning = f"ignored magnets (tilt must be -90..90 degrees): {'; '.join(rejected)}" if rejected else ""
        self.status.error = self.warning
        self.last_magnet_visit = 0.0
        self.move_log: list[tuple[float, float]] = []
        self._arrived_at = 0.0  # when the camera reached the current view
        self._windows_at = -1e18  # when one of the camera's own windows last reached the gateway
        self._clean_window_at = -1e18  # when one was last answered for the current view (held still throughout)
        self._window_yes_until = 0.0  # a window said yes: keep holding until then
        self._last_look_p = 0.0
        self._last_jpeg = b""
        self.time_wide = self.time_zoomed = 0.0  # recent seconds at wide and at zoomed views (decaying)
        self._stop = threading.Event()
        self._return_home = True
        self._thread: threading.Thread | None = None

    # control
    def start(self) -> None:
        self._thread = threading.Thread(target=self.run, daemon=True, name=f"hunter-{self.config.camera_id[:8]}")
        self._thread.start()

    def stop(self, wait: bool = True, return_home: bool = True) -> None:
        """Stop hunting. With ``return_home`` False the motor just stops where it is (a replacement hunter takes
        over, and two hunters must never drive the same camera)."""
        self._return_home = return_home
        self._stop.set()
        if not return_home:
            try:
                self.ptz.continuous(0.0, 0.0, 0.0)
            except Exception:  # noqa: BLE001
                pass
        if wait and self._thread:
            self._thread.join(timeout=300)

    def _pause(self, seconds: float) -> bool:
        """Sleep; False if stopped meanwhile."""
        if self.sleep is time.sleep:
            return not self._stop.wait(seconds)
        self.sleep(seconds)
        return not self._stop.is_set()

    def _load(self) -> View | None:
        """Views, priors and finds from earlier runs, so what it learned survives restarts. Returns the remembered home."""
        if not (self.memory and self.memory.exists()):
            return None
        data = json.loads(self.memory.read_text())
        current = isinstance(data, dict) and data.get("version") == MEMORY_VERSION
        for row in data.get("views", []) if current else []:  # older views: scored by other prompts, re-survey
            self.stats[View(row["pan"], row["tilt"], row["fov"])] = ViewStats.from_json(row["stats"])
        home = data.get("home") if isinstance(data, dict) else None
        return View(home["pan"], home["tilt"], home["fov"]) if home else None

    def _save(self) -> None:
        if not self.memory:
            return
        rows = [{"pan": v.pan, "tilt": v.tilt, "fov": v.fov, "stats": s.to_json()} for v, s in self.stats.items()]
        home = vars(self.home) if self.home else None
        tmp = self.memory.with_suffix(".tmp")
        tmp.write_text(json.dumps({"version": MEMORY_VERSION, "home": home, "views": rows}))
        tmp.replace(self.memory)

    def hot_spots(self, n: int = 5) -> list[tuple[View, ViewStats]]:
        return sorted(((v, s) for v, s in self.stats.items() if s.total_hits), key=lambda vs: -vs[1].total_hits)[:n]

    def moved_during(self, start: float, end: float) -> bool:
        """Did the hunter move the camera at any time in [start, end]? Such windows show camera motion."""
        return any(s < end and e > start for s, e in self.move_log) or self.status.state == "returning"

    # the loop
    def run(self) -> None:
        fatal = False
        try:
            self.status.state = "waiting for PTZ"
            while True:  # the Server may still be starting; never move before the PTZ answers
                try:
                    current = self.ptz.position()
                    break
                except Exception as error:  # noqa: BLE001
                    self.status.error = f"waiting for PTZ: {error!r}"[:200]
                    if not self._pause(10.0):
                        return
            self.status.error = self.warning
            remembered = self._load()
            # Home is the camera's normal view, not wherever an earlier run left it: the home preset if set, else the
            # first home this hunter ever saw (remembered), else where the camera is now.
            located = self.locate_preset(self.config.home_preset) if (self.config.home_preset and self.locate_preset) else None
            self.home = located or remembered or current
            self._save()
            wide = max(self.home.fov, current.fov)
            self.home = View(self.home.pan, self.home.tilt, self.home.fov or wide)
            for preset in self.config.presets:
                located = self.locate_preset(preset) if self.locate_preset else None
                if located is not None:
                    self.magnets.append(View.of(located.pan, located.tilt, located.fov or wide))
            self.magnets = [View.of(m.pan, m.tilt, m.fov or wide) for m in self.magnets]
            home = View.of(self.home.pan, self.home.tilt, self.home.fov)
            if all(home.distance(m) > 5.0 for m in self.magnets):
                self.magnets.append(home)
            for magnet in self.magnets:
                self.stats.setdefault(magnet, ViewStats(prior=0.7))
            self.last_magnet_visit = self.clock()
            self.status.state = "surveying"
            for view in survey_views(self.config, self.home, self.magnets):
                if self._stop.is_set():
                    break
                if self.stats.get(view) and self.stats[view].visits:
                    continue  # surveyed in an earlier run
                self.stats.setdefault(view, ViewStats())
                try:
                    self._visit(view, survey=True)
                except (OSError, ValueError) as error:  # the Server is unreachable (restarting, say)
                    if not self._recover(error):
                        break
            self.status.state = "hunting"
            while not self._stop.is_set():
                hour = datetime.fromtimestamp(self.clock()).hour
                visit_started = self.clock()
                view = choose_next(self._candidates(), self.status.view, self.clock(), hour, magnets=self.magnets,
                                   magnet_weight=self.config.magnet_weight, magnet_radius=self.config.magnet_radius,
                                   since_magnet_s=self.clock() - self.last_magnet_visit,
                                   magnet_return_s=self.config.magnet_return_s)
                try:
                    self._visit(view, survey=False)
                except (OSError, ValueError) as error:  # the Server is unreachable (restarting, say)
                    if not self._recover(error):
                        break
                self._spent(self.status.view or view, self.clock() - visit_started)
        except Exception as error:  # noqa: BLE001 - report, then hand the camera back
            fatal, self.status.error = True, repr(error)
        finally:
            if self.home is not None and self._return_home:
                self.status.state = "returning"
                try:
                    move_slowly(self.ptz, self.home, self.config.max_speed, sleep=self.sleep)
                except Exception as error:  # noqa: BLE001
                    fatal, self.status.error = True, f"return home failed: {error!r}"
            self.status.state = "error" if fatal else "stopped"

    def _is_wide(self, view: View) -> bool:
        return self.home is None or view.fov >= 0.9 * self.home.fov

    def _spent(self, view: View, seconds: float, half_life_s: float = 1800.0) -> None:
        """Count time at wide and at zoomed views, the last half hour or so weighing most."""
        decay = 0.5 ** (max(0.0, seconds) / half_life_s)
        self.time_wide *= decay
        self.time_zoomed *= decay
        if self._is_wide(view):
            self.time_wide += seconds
        else:
            self.time_zoomed += seconds

    def _candidates(self) -> dict[View, ViewStats]:
        """Wide views while the hunter has spent less than ``wide_share`` of its recent time wide; otherwise it gets
        bored and looks closer, at zoomed views (all views when there are none of the kind wanted)."""
        total = self.time_wide + self.time_zoomed
        want_wide = total <= 0.0 or self.time_wide / total < self.config.wide_share
        chosen = {v: s for v, s in self.stats.items() if self._is_wide(v) == want_wide}
        return chosen if any(s.avoid < 0.8 for s in chosen.values()) else self.stats

    def _recover(self, error: Exception) -> bool:
        """The Server stopped answering mid-hunt: stop the motor if it can, wait until the PTZ answers again, and
        carry on. False if stopped meanwhile."""
        self.status.error = f"Server unreachable, waiting: {error!r}"[:200]
        try:
            self.ptz.continuous(0.0, 0.0, 0.0)
        except Exception:  # noqa: BLE001 - it is unreachable; nothing more to do
            pass
        while self._pause(10.0):
            try:
                self.ptz.position()
            except Exception:  # noqa: BLE001
                continue
            self.status.error = self.warning
            self.status.view = None  # where it is now is unknown: the next move starts from the real position
            return True
        return False

    def _go(self, view: View) -> View | None:
        """Move there slowly, logging the move so windows during it can be skipped. Returns the view the camera is
        really at: ``view`` if it got there, else where it ended up (what it sees is learned for that view, not for
        one it never reached). None if stopped."""
        if self.status.view is not None and view.distance(self.status.view) > 1.0:
            if not self._pause(self.config.rest_s):
                return None
        target = View(view.pan, clamp_tilt(self.config, view.tilt), view.fov)
        started, before = self.clock(), self.ptz.position()
        self.move_log.append((started, float("inf")))
        zoomed = False
        try:
            after, zoomed = self._move(before, target)
        finally:
            # Refocusing after a zoom blurs the picture: windows over it are camera motion too.
            self.move_log[-1] = (started, self.clock() + self.config.settle_s + (self.config.focus_s if zoomed else 0.0))
        if before.distance(after) <= 1.0:
            self.move_log.pop()  # already there
        else:
            self.status.moves += 1
        self.move_log = self.move_log[-50:]
        arrived = view if after.distance(view) <= max(3.0, 0.1 * view.fov) else View.of(after.pan, after.tilt, after.fov)
        if arrived != self.status.view:
            self._arrived_at = self.clock()
        self.status.view = arrived
        return arrived if self._pause(self.config.settle_s + (self.config.focus_s if zoomed else 0.0)) else None

    def _move(self, before: View, target: View) -> tuple[View, bool]:
        """Pan and tilt slowly (the motors are loud); zoom in one quick step (it is silent): out before panning, so the
        sweep is wide, and in after it, once the spot is centred. Returns (where it is, whether the zoom changed)."""
        zoom = abs(target.fov - before.fov) > FOV_TOL
        if zoom and target.fov > before.fov:
            self.ptz.absolute(View(before.pan, before.tilt, target.fov))
            self.sleep(1.0)
        now = self.ptz.position()
        after = move_slowly(self.ptz, View(target.pan, target.tilt, now.fov), self.config.max_speed, sleep=self.sleep)
        if zoom and target.fov < before.fov:
            self.ptz.absolute(View(after.pan, after.tilt, target.fov))
            self.sleep(1.0)
            after = self.ptz.position()
        return after, zoom

    def _ask(self, jpeg: bytes | list[bytes], questions: dict) -> Answers | None:
        try:
            answers = self.ask(self.config.camera_id, jpeg, questions)
        except Exception as error:  # noqa: BLE001 - e.g. dropped under load: carry on
            self.status.error = f"skipped: {error!r}"[:300]
            return None
        self.status.error = self.warning
        return answers

    def _still_at(self, view: View) -> bool:
        """Is the camera still where the hunter left it? Something else may have moved it: the Server drives a camera
        with a PTZ home position back home after a few idle minutes (while Clef is busy, say), or an operator."""
        try:
            now = self.ptz.position()
        except Exception:  # noqa: BLE001 - cannot tell: assume it is
            return True
        if now.distance(view) <= max(3.0, 0.1 * view.fov):
            return True
        self.status.error = f"camera was moved to pan {now.pan:.1f} tilt {now.tilt:.1f}: answers dropped"
        self.status.view = View.of(now.pan, now.tilt, now.fov)
        return False

    def _look(self, survey: bool, view: View) -> tuple[float, str, tuple | None] | None:
        """Ask the frame (and tiles); return (best p, question, tile box or None) or None if not answered, or if the
        camera was no longer at ``view`` when the snapshot was taken (the answers would not be about this view; once
        taken, the snapshot stays valid however long Clef takes)."""
        cfg = self.config
        jpeg = self.ptz.snapshot()
        if not self._still_at(view):
            return None
        hunt = {f"t{i}": noul(text) for i, text in enumerate(cfg.targets)}
        questions = dict(hunt)
        stats = self.stats.setdefault(view, ViewStats())
        self._last_jpeg = jpeg
        if survey or not stats.scored or (cfg.cues and not stats.cues):
            questions.update(likelihood_questions(cfg.targets, cfg.interests, cfg.avoid, cfg.cues))
        answers = self._ask(jpeg, questions)
        if answers is None:
            stats.last_visit = self.clock()
            return None
        if any(q.startswith("_") for q in answers):
            stats.scene, stats.base_avoid = prior_from(answers)
            extra = list(cfg.avoid)
            stats.extra_avoid = {extra[int(q[6:]) - len(DEFAULT_AVOID)]: a["true"] for q, a in answers.items()
                                 if q.startswith("_avoid") and "true" in a
                                 and 0 <= int(q[6:]) - len(DEFAULT_AVOID) < len(extra)}
            stats.cues = {cfg.cues[int(q[4:])]: a["true"] for q, a in answers.items()
                          if q.startswith("_cue") and "true" in a and int(q[4:]) < len(cfg.cues)}
            stats.scored = True
            self._reprice()
        self.status.last_answers = {q: round(a["true"], 3) for q, a in answers.items() if "true" in a}
        best = max(((answers[q]["true"], cfg.targets[int(q[1:])], None) for q in hunt), key=lambda b: b[0])
        self._last_look_p = best[0]
        boxes = tile_boxes(cfg.tiles) if stats.avoid < 0.8 else []
        if boxes:
            tiles = [self.crop(jpeg, *box) for box in boxes]
            # Clef says well whether something is in any of several images, not which: ask all the tiles at once, and
            # one by one only to find where (an empty view, the usual case, costs one request instead of nine).
            gate = self._ask(tiles, {q: noul("These images are overlapping tiles of one camera snapshot. "
                                             f"{cfg.targets[int(q[1:])]} Answer yes if so in any of them.")
                                     for q in hunt})
            if gate:
                self._last_look_p = max(self._last_look_p, *(gate[q]["true"] for q in hunt))
            if gate and max(gate[q]["true"] for q in hunt) >= cfg.tile_gate:
                for box, tile in zip(boxes, tiles):
                    tile_answers = self._ask(tile, hunt)
                    if tile_answers:
                        for q in hunt:
                            p = tile_answers[q]["true"]
                            if p > best[0]:
                                best = (p, cfg.targets[int(q[1:])], box)
        self._last_look_p = max(self._last_look_p, best[0])
        return best

    def _visit(self, view: View, survey: bool) -> None:
        cfg = self.config
        view = self._go(view)
        if view is None:
            return
        if any(view.distance(m) <= 5.0 for m in self.magnets):
            self.last_magnet_visit = self.clock()
        result = self._look(survey, view)
        stats = self.stats.setdefault(view, ViewStats())
        now, hour = self.clock(), datetime.fromtimestamp(self.clock()).hour
        stats.visits += 1
        stats.last_visit = now
        stats.checks[hour] = stats.checks.get(hour, 0) + 1
        if result is None:
            return
        stats.observe(hour, self._last_look_p)
        p, question, box = result
        if box is not None and cfg.zoom_in and p >= cfg.curious:
            # Something small, or something the whole frame was unsure about: centre on it and zoom in to look.
            phase = self.status.state
            self.status.state = "zooming"
            closer = self._go(aim_at(view, box, cfg.min_fov))
            if closer is None:
                return
            answers = self._ask(self.ptz.snapshot(), {"t": noul(question)})
            p = answers["t"]["true"] if answers else 0.0
            spot = self.stats.setdefault(closer, ViewStats(prior=stats.prior))
            spot.visits += 1
            spot.last_visit = self.clock()
            spot.checks[hour] = spot.checks.get(hour, 0) + 1
            if answers:
                spot.observe(hour, p)
            if p >= cfg.threshold:
                # The zoomed view is the hot spot to come back to, not just the wide one it was found from.
                spot.hits[hour] = spot.hits.get(hour, 0) + 1
                spot.last_yes = self.clock()
                view = closer
            self.status.state = phase
        if p >= cfg.threshold:
            stats.hits[hour] = stats.hits.get(hour, 0) + 1
            stats.last_yes = now
            self._save()
            self._dwell(view, question, p)
        self._save()
        if not survey and self._worth_holding(view):
            self._scan_tiles(view, self._last_jpeg)
            self._hold()
        if not survey and stats.prior >= 0.6:  # explore around promising views
            half = view.fov * cfg.overlap / 2
            for pan in (view.pan - half, view.pan + half):
                self.stats.setdefault(View.of(pan, view.tilt, view.fov), ViewStats(prior=stats.prior * 0.9))

    def _reprice(self) -> None:
        """Each view's prior and avoid: its scene score, raised by helper answers and lowered by the operator's avoid
        questions, each only as far as it stands out from this camera's usual answer (the median over its views):
        "Is there a tree in view?" is yes in almost every view of a yard, and must not rule them all out."""
        cue_medians = camera_medians([s.cues for s in self.stats.values() if s.cues])
        avoid_medians = camera_medians([s.extra_avoid for s in self.stats.values() if s.extra_avoid])
        for s in self.stats.values():
            if s.scene < 0.0:
                continue
            extra = max([max(0.0, p - avoid_medians.get(q, 0.0)) / max(0.05, 1.0 - avoid_medians.get(q, 0.0))
                         for q, p in s.extra_avoid.items()] + [0.0])
            s.avoid = max(s.base_avoid, extra)
            s.prior = s.scene * (1.0 - extra) * (cue_boost(s.cues, cue_medians) if s.cues else 0.75)

    def _scan_tiles(self, view: View, jpeg: bytes) -> None:
        """Ask each tile of a promising view the helper questions (at most every ``CUE_RESCAN_S``), and add a zoomed
        view on each tile where one stands out: the hunter then looks closer where a yes is likelier. Wide views and
        once-zoomed views are scanned (two levels: about 18 and 9 degrees from a 37 degree wide view)."""
        cfg, stats = self.config, self.stats.get(view)
        if (not cfg.cues or not cfg.zoom_in or not jpeg or stats is None or self.home is None
                or view.fov < 0.4 * self.home.fov or self.clock() - stats.cue_scan_at < CUE_RESCAN_S):
            return
        boxes = tile_boxes(cfg.tiles)
        questions = {f"c{i}": noul(text) for i, text in enumerate(cfg.cues)}
        tiles = []
        for box in boxes:
            answers = self._ask(self.crop(jpeg, *box), questions)
            if answers is None:
                return  # Clef too busy: try again next time
            tiles.append({cfg.cues[int(q[1:])]: a["true"] for q, a in answers.items()})
        stats.cue_scan_at = self.clock()
        for index, score, cue in standout_tiles(tiles)[:ZOOM_CHILDREN]:
            child = zoom_on(view, boxes[index], cfg.min_fov)
            if any(v.fov < view.fov and v.distance(child) < 0.25 * child.fov for v in self.stats):
                continue  # already a zoomed view there
            self.stats[child] = ViewStats(prior=stats.prior * (0.5 + 0.5 * score), found_by=cue)
        self._save()

    # The camera's own windows (from the plugin, through the gateway)
    def window_arrived(self) -> None:
        """One of the camera's own windows reached the gateway (answered or not): they are coming, so hold for one."""
        self._windows_at = self.clock()

    def observe_window(self, start: float, end: float, p: float) -> None:
        """An answered window of the camera's own questions. If the camera held the current view throughout, it is a
        look at that view (with ``p``, the strongest yes), and a yes keeps the hunter there."""
        view = self.status.view
        if view is None or view not in self.stats or self.moved_during(start, end):
            return
        if start < self._arrived_at + self.config.settle_s or self.status.state not in ("hunting", "holding", "dwelling"):
            return
        self.stats[view].observe(datetime.fromtimestamp(end).hour, p)
        self._clean_window_at = self.clock()
        if p >= self.config.threshold:
            self._window_yes_until = self.clock() + self.config.dwell_s

    def _worth_holding(self, view: View, share: float = 0.5) -> bool:
        """Hold only views at least half as promising as the best one now; elsewhere look once and move on."""
        hour = datetime.fromtimestamp(self.clock()).hour
        best = max([s.yield_rate(hour) for s in self.stats.values() if s.avoid < 0.8] + [1e-9])
        here = self.stats.get(view)
        return here is not None and here.yield_rate(hour) >= share * best

    def _hold(self) -> None:
        """Stay until one of the camera's own windows has been answered for this view (``hold_s`` at most), and as
        long as its answers say yes (``max_dwell_s`` at most). Windows overlapping a move are dropped, so without this
        the camera's own questions were never answered while hunting. Nothing to wait for if no windows come."""
        cfg = self.config
        if self.clock() - self._windows_at > 120.0:
            return
        phase = self.status.state
        self.status.state = "holding"
        while not self._stop.is_set():
            held = self.clock() - self._arrived_at
            if self.clock() < self._window_yes_until and held < cfg.max_dwell_s:
                pass
            elif self._clean_window_at >= self._arrived_at or held >= cfg.hold_s:
                break
            if not self._pause(1.0):
                break
        self.status.state = phase

    def learned(self, hour: int) -> list[dict]:
        """Every scored view with its yield now, relative to the best, and whether it is quiet."""
        best = max([s.yield_rate(hour) for s in self.stats.values() if s.avoid < 0.8] + [1e-9])
        seen_anywhere = sightings(self.stats)
        rows = [{"view": vars(v), "yield": round(s.yield_rate(hour) / best, 3), "looks": s.total_looks,
                 "quiet": is_quiet(s, hour, best, seen_anywhere), "avoid": round(s.avoid, 2),
                 **({"zoomed_for": s.found_by} if s.found_by else {})}
                for v, s in self.stats.items()]
        return sorted(rows, key=lambda r: -r["yield"])

    def _dwell(self, view: View, question: str, probability: float) -> None:
        """Stay while the answer stays yes, so the camera's normal windows catch it (events, best shots)."""
        cfg = self.config
        phase = self.status.state
        self.status.state, self.status.found = "dwelling", f"{question} ({probability:.2f})"
        if self.on_found:
            self.on_found(cfg, view, question, probability)
        started = self.clock()
        until = started + cfg.dwell_s
        while self.clock() < min(until, started + cfg.max_dwell_s) and not self._stop.is_set():
            if not self._pause(min(15.0, cfg.dwell_s)):
                break
            jpeg = self.ptz.snapshot()
            if not self._still_at(view):
                break
            answers = self._ask(jpeg, {"t": noul(question)})
            if answers and answers["t"]["true"] >= cfg.threshold:
                until = self.clock() + cfg.dwell_s
        self.status.state, self.status.found = phase, ""
