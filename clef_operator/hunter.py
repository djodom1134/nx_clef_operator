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

Pan and tilt are always slow: continuous mode at a capped speed, axes timed to arrive together, a full stop on arrival,
a rest between moves, and slower passes for the last few degrees (to within 2). Absolute moves ignore their speed and are
loud, so they are used only for zoom, which is silent. An axis that stops moving while driven is at its end stop (on Back PTZ,
tilting up into the end stop makes the camera pan by itself). Views are recorded where the camera actually ended up.
"""

from __future__ import annotations

import json
import math
import subprocess
import tempfile
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Protocol

from . import embed as embedding
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
MEMORY_VERSION = 3  # older than 2: forgotten on load (other prompts, or positions never reached); 2: see _load


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
    cycle_s: float = 240.0  # boredom cycle: wide_share of it watching wide views, then it looks around close up
    bored_weight: float = 0.6  # how much recently watched areas (not just views) count against going back there
    handback_s: float = 10.0  # someone else moved the camera (an operator, the camera's own auto-tracking): hand it
    #                            over, and take it back only after it has been still this long
    keepalive_s: float = 60.0  # while still, a stop command this often: the Server sends an idle PTZ to its home
    #                            position at full speed (loud), so it must never look idle while the hunter runs
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
HOPELESS_BLOCKED, HOPELESS_AVOID, HOPELESS_SCENE = 0.5, 0.7, 0.15  # see ``hopeless``


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
    embed: float = -1.0  # EmbeddingGemma's good-place share of the view (-1: not embedded; see embed.py)

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
                "extra_avoid": self.extra_avoid, "embed": self.embed}

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
                         d.get("extra_avoid", {}), d.get("embed", -1.0))


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


def hopeless(s: ViewStats) -> bool:
    """No chance of a yes here: the scene questions said it is mostly wall, roof or sky (the built-in blockers at
    ``HOPELESS_BLOCKED`` or more), or little open ground (scene score under ``HOPELESS_SCENE``), or the avoid answers
    together reach ``HOPELESS_AVOID``, or EmbeddingGemma rates it below ``embed.HOPELESS`` (then Clef's scene
    questions are not even asked) - and the target has never been found here. The hunter never goes there to hunt
    (only the survey looks once); a soft avoid penalty alone let boredom push it onto the side of the house once the
    good views had all been watched."""
    if s.total_hits:
        return False
    if 0.0 <= s.embed < embedding.HOPELESS:  # EmbeddingGemma: far more like a wall, roof, sky or leaves than a yard
        return True
    if not s.scored:
        return False
    return s.base_avoid >= HOPELESS_BLOCKED or s.avoid >= HOPELESS_AVOID or 0.0 <= s.scene < HOPELESS_SCENE


def is_quiet(s: ViewStats, hour: int, best_yield: float, seen_anywhere: float = SIGHTINGS) -> bool:
    return (seen_anywhere >= SIGHTINGS and s.total_looks >= QUIET_LOOKS
            and s.yield_rate(hour) < QUIET_SHARE * best_yield)


def resting(s: ViewStats, now: float, hour: int, best_yield: float, seen_anywhere: float) -> bool:
    """A quiet view is skipped until its recheck is due."""
    return is_quiet(s, hour, best_yield, seen_anywhere) and now - s.last_visit < QUIET_RECHECK_S


def choose_next(stats: dict[View, ViewStats], current: View | None, now: float, hour: int,
                travel_weight: float = 0.002, exploration: float = 0.15, magnets: list[View] | None = None,
                magnet_weight: float = 0.5, magnet_radius: float = 25.0, since_magnet_s: float = 0.0,
                magnet_return_s: float = 600.0, hot_weight: float = 0.8, hot_revisit_s: float = 300.0,
                recent: list[tuple[float, View]] | None = None, bored_weight: float = 0.6,
                bored_half_life_s: float = 600.0, bored_cap: float = 4.0) -> View:
    """Most promising view: learned yield (relative to the best view) and hits, skipping quiet views, plus staleness, exploration, magnet pull and hot-spot pull, minus
    travel (per degree). The magnet pull grows the longer the hunter has been away from every magnet. Hot spots are
    views where the target was found: the more finds, the stronger the pull, and it comes back within minutes of a
    visit (``hot_revisit_s``) instead of the half hour ordinary views take to go stale. Travel costs little (0.2 for
    100 degrees, about 15 s of moving): at 0.01 a degree the best views across the yard were never worth the trip, and
    the hunter bounced between two neighbouring views for hours.

    Avoid answers below the veto (0.8) still count: a view's value is scaled by (1 - avoid). Boredom: each recently
    watched spot (``recent``: (time, view)) counts against views around it (within about half a field of view),
    fading with ``bored_half_life_s``, so the hunter moves on to other good areas instead of circling one. Each visit
    scales a view's appeal by (1 - ``bored_weight``), up to ``bored_cap`` visits' worth. Subtracted and capped at one
    visit, boredom never outweighed a hot spot's pull plus the home magnet's (the hunter circled its home area); and
    subtracted at all, it pushed the hunter onto walls once the good views were all bored. Scaled, a view with no
    promise never wins, and hopeless views (``hopeless``) are never chosen."""
    most_hits = max([s.total_hits for s in stats.values()] + [1])
    total_visits = sum(s.visits for s in stats.values()) + 1
    pull_strength = magnet_weight * (0.1 + min(1.0, since_magnet_s / magnet_return_s)) if magnets else 0.0
    best_yield = max([s.yield_rate(hour) for s in stats.values() if s.avoid < 0.8 and not hopeless(s)] + [1e-9])
    seen_anywhere = sightings(stats)

    def score(view: View) -> float:
        s = stats[view]
        if s.avoid >= 0.8 or hopeless(s):
            return -2000.0
        value = s.yield_rate(hour) / best_yield * (1.0 - s.avoid)  # relative to the best view; avoid answers count
        if resting(s, now, hour, best_yield, seen_anywhere):
            return -1000.0  # below any active view, however bored (all tied at -0.5, quiet home won every tie)
        stale = min(1.0, (now - s.last_visit) / 1800.0) if s.last_visit else 1.0
        # Curiosity in proportion to promise: an unvisited wall is not worth a look just for being unvisited.
        explore = exploration * value * math.sqrt(math.log(total_visits) / (s.visits + 1))
        travel = travel_weight * (view.distance(current) if current else 0.0)
        repeat = 0.5 if current and view.distance(current) <= 3.0 else 0.0  # near copies count as staying put
        pull = pull_strength * magnet_pull(view, magnets or [], magnet_radius) * (1.0 - s.avoid)  # home's area, not its wall
        hot = 0.0
        if s.total_hits:
            fame = math.log1p(s.total_hits) / math.log1p(most_hits)
            due = min(1.0, (now - s.last_visit) / hot_revisit_s) if s.last_visit else 1.0
            hot = hot_weight * fame * due
        bored = min(bored_cap, sum(
            0.5 ** ((now - t) / bored_half_life_s) * math.exp(-(view.distance(v) / (0.5 * max(view.fov, v.fov))) ** 2)
            for t, v in (recent or ())))
        appeal = value * (0.5 + 0.5 * stale) + explore + pull + hot
        return appeal * (1.0 - bored_weight) ** bored - travel - repeat  # bored scales appeal: a hopeless view never wins

    return max(stats, key=score)


def same_spot(a: View, b: View, share: float = 0.25) -> bool:
    """Close enough to be the same view: within ``share`` of the field of view, at about the same zoom."""
    return abs(a.fov - b.fov) <= share * max(a.fov, b.fov) and a.distance(b) <= share * max(a.fov, b.fov)


def merge_stats(into: ViewStats, other: ViewStats) -> None:
    """Fold a near-duplicate view's history into another (the more visited one keeps its scores)."""
    into.visits += other.visits
    for mine, theirs in ((into.hits, other.hits), (into.checks, other.checks), (into.looks, other.looks)):
        for hour, n in theirs.items():
            mine[hour] = mine.get(hour, 0) + n
    for hour, p in other.seen.items():
        into.seen[hour] = into.seen.get(hour, 0.0) + p
    into.last_visit = max(into.last_visit, other.last_visit)
    into.last_yes = max(into.last_yes, other.last_yes)
    into.cue_scan_at = max(into.cue_scan_at, other.cue_scan_at)
    into.found_by = into.found_by or other.found_by


def merge_duplicates(stats: dict[View, ViewStats]) -> dict[View, ViewStats]:
    """One view per spot: near-duplicates (moves that ended a little off, neighbours added twice) merged into the most
    visited of them, so what is learned about a spot is not spread over a dozen copies."""
    kept: dict[View, ViewStats] = {}
    for view, s in sorted(stats.items(), key=lambda vs: -vs[1].visits):
        twin = next((k for k in kept if same_spot(k, view)), None)
        if twin is None:
            kept[view] = s
        else:
            merge_stats(kept[twin], s)
    return kept


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


class ExternalControl(Exception):
    """Something other than the hunter is moving the camera (an operator in Nx, or the camera's own auto-tracking)."""

    def __init__(self, where: View, unstopped: dict[str, float] | None = None, why: str = "") -> None:
        super().__init__(f"camera moved by someone else to pan {where.pan:.1f} tilt {where.tilt:.1f}")
        self.where, self.why = where, why
        self.unstopped = unstopped or {}  # the hunter's own command, left running so as not to stop theirs


def obeys(speeds: dict[str, float], before: View, after: View, seconds: float) -> bool | None:
    """Is the camera moving the way the hunter's continuous command ``speeds`` drives it (so the command is still in
    force, and nobody else's replaced it)? True if every driven pan/tilt axis moves in the commanded direction at
    10-160% of the commanded rate (Back PTZ slows pan and tilt as it zooms in: at 17 degrees of field of view, speed
    0.2 pans about 2 degrees a second, not 6; an axis held at an end stop may stand still, but not all of them); False
    if it moves some other way; None if it stands still."""
    moved = {"pan": angle_delta(before.pan, after.pan) / seconds, "tilt": (after.tilt - before.tilt) / seconds}
    if all(abs(v) < 0.3 for v in moved.values()):
        return None
    following = False
    for axis in ("pan", "tilt"):
        expected = speeds.get(axis, 0.0) * RATE[axis]
        if abs(expected) < 0.3:
            if abs(moved[axis]) > max(1.0, 0.5 * max(abs(moved[a]) for a in ("pan", "tilt"))):
                return False  # an axis the hunter does not drive moves by itself (beyond end-stop creep)
            continue
        if abs(moved[axis]) < 0.3:
            continue  # held at an end stop
        if math.copysign(1.0, moved[axis]) != math.copysign(1.0, expected) or not 0.1 <= abs(moved[axis] / expected) <= 1.6:
            return False
        following = True
    return following


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
    taken_over = False
    seen: list[tuple[float, View]] = []
    try:
        while elapsed < limit:
            sleep(0.25)
            elapsed += 0.25
            now = ptz.position()
            seen.append((elapsed, now))
            left = _left(now, target)
            for axis in driven:
                unchanged[axis] = unchanged[axis] + 0.25 if abs(left[axis] - last[axis]) < 0.01 else 0.0
                last[axis] = left[axis]
                if unchanged[axis] >= STALL_S:
                    stalled.add(axis)
            # An axis the hunter is not driving moved: someone else has the camera, unless a driven axis has stalled
            # (tilting into the end stop makes Back PTZ pan by itself).
            # Confirmed only once the driven axes are seen not to follow the hunter's command (over 1.5 s: the readback
            # comes about once a second): a zoom readback still settling is not an operator.
            if any(abs(left[a] - remaining[a]) > (3.0 if a != "fov" else 4.0) for a in ("pan", "tilt", "fov") if a not in driven) \
                    and not any(unchanged[a] >= 0.25 for a in driven):
                back = next((v for t, v in reversed(seen) if elapsed - t >= 1.5), None)
                pan_tilt = {a: speeds[a] for a in ("pan", "tilt") if speeds[a]}
                pan_or_tilt_moved = any(abs(left[a] - remaining[a]) > 3.0 for a in ("pan", "tilt") if a not in driven)
                if back is not None and (pan_or_tilt_moved or (pan_tilt and obeys(pan_tilt, back, now, 1.5) is False)):
                    taken_over = True
                    moved = ", ".join(f"{a} {left[a] - remaining[a]:+.1f}" for a in ("pan", "tilt", "fov") if a not in driven)
                    raise ExternalControl(now, {a: speeds[a] for a in driven},
                                          f"mid-move driving {'+'.join(driven)}: undriven {moved}")
            if stalled or any(abs(left[a]) <= lead[a] or math.copysign(1.0, left[a]) != direction[a]
                              or abs(left[a]) > abs(remaining[a]) + 2.0 for a in driven):
                break
    finally:
        if not taken_over:  # never stop someone else's move: their command already replaced the hunter's
            ptz.continuous(0.0, 0.0, 0.0)
    return stalled


APPROACH_SPEED = 0.06  # within 10 degrees: about 1.5 degrees a second of pan, so a pass overshoots by about one
CLOSE_ENOUGH = 2.0  # degrees of pan or tilt: no more correcting


def move_slowly(ptz: Ptz, target: View, max_speed: float, sleep: Callable[[float], None] = time.sleep,
                timeout_s: float = 300.0, passes: int = 6, close_enough: float = CLOSE_ENOUGH) -> View:
    """Drive toward ``target`` at no more than ``max_speed`` and stop within ``close_enough`` degrees. Every move is a
    slow continuous move, never an absolute one: cameras ignore the speed of absolute moves, and the jump is loud
    (going back to a preset was audible). Each pass stops a little early (the Server reports positions late); within
    10 degrees, passes slow to ``APPROACH_SPEED``. An axis that stalled at an end stop is left where it is. Returns
    where the camera really is (the caller learns each view there)."""
    stuck: set[str] = set()
    for i in range(passes + 1):
        if i:
            sleep(1.0)  # let the readback catch up before judging the last pass
        now = ptz.position()
        left = _left(now, target)
        limits = {"pan": close_enough, "tilt": close_enough, "fov": 2 * FOV_TOL}
        if i == passes or all(abs(left[a]) <= limits[a] or a in stuck for a in left):
            return now
        goal = View(now.pan if "pan" in stuck else target.pan, now.tilt if "tilt" in stuck else target.tilt,
                    now.fov if "fov" in stuck else target.fov)
        far = max([abs(left[a]) for a in ("pan", "tilt") if a not in stuck] + [0.0])
        stuck |= _pass(ptz, goal, max_speed if far > 10.0 else min(max_speed, APPROACH_SPEED), sleep, timeout_s)
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
                 memory: Path | None = None, embedder: "embedding.Embedder | None" = None, embed_map=None) -> None:
        self.config, self.ptz, self.ask, self.on_found = config, ptz, ask, on_found
        self.embedder = embedder  # cheap looks (embed.py); None: Clef does everything
        self.embed_map = embed_map  # what each direction looked like, searchable with text (embed_map.py)
        self._look_vectors: tuple[bytes, list[list[float]]] | None = None  # the last look's snapshot, its tile vectors
        self._requested: View | None = None  # "look there", from the dashboard's search
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
        self._ptz_lock = threading.Lock()  # moves and keep-alive commands never overlap
        self._last_command = time.time()
        self._quiet_until = 0.0  # no commands before this (the camera is refocusing)
        self._expected: View | None = None  # where the hunter left the camera; elsewhere = someone else moved it
        self._runaway: tuple[float, View, dict[str, float], int] | None = None  # see _still_ours
        self._steady: tuple[float, tuple[float, float]] | None = None  # see _steady_too_long
        self._handed_over_until = 0.0  # someone else has the camera until then (extended while it keeps moving)
        self.recent: deque = deque(maxlen=60)  # (time, view) watched, for boredom
        self._mode, self._mode_until = "wide", None  # boredom cycle: "wide" watching, or "looking around" close up
        self._stop = threading.Event()
        self._return_home = True
        self._thread: threading.Thread | None = None

    # control
    def start(self) -> None:
        self._thread = threading.Thread(target=self.run, daemon=True, name=f"hunter-{self.config.camera_id[:8]}")
        self._thread.start()
        threading.Thread(target=self._keepalive, daemon=True, name=f"hunter-keepalive-{self.config.camera_id[:8]}").start()

    def _keepalive(self) -> None:
        """Every 2 s while the camera is still: notice if someone else moved it (then hand it over), and every
        ``keepalive_s`` send a stop command, so the Server never sees an idle PTZ and swings it home at full speed."""
        while not self._stop.wait(2.0):
            now = time.time()
            if not self._watch(now):
                self._keepalive_step(now)

    def _watch(self, now: float) -> bool:
        """True if someone else is (or was lately) moving the camera. Not checked during the hunter's own moves or
        while it refocuses."""
        if now < self._quiet_until or self._expected is None:
            return now < self._handed_over_until
        if not self._ptz_lock.acquire(blocking=False):  # the hunter is moving it
            return now < self._handed_over_until
        try:
            where = self.ptz.position()
        except Exception:  # noqa: BLE001 - the hunter's own loop reports Server trouble
            return now < self._handed_over_until
        finally:
            self._ptz_lock.release()
        if self._runaway and self._still_ours(now, where):
            return False
        if now < self._handed_over_until and self._steady_too_long(now, where):
            return False
        if abs(angle_delta(where.pan, self._expected.pan)) > 2.5 or abs(where.tilt - self._expected.tilt) > 2.5 \
                or abs(where.fov - self._expected.fov) > max(2.0, 0.1 * self._expected.fov):
            self._handed_over(now, where, why=f"watcher: expected pan {self._expected.pan:.1f} tilt "
                                              f"{self._expected.tilt:.1f} fov {self._expected.fov:.1f}")
        return now < self._handed_over_until

    def _steady_too_long(self, now: float, where: View, limit_s: float = 60.0) -> bool:
        """Last resort while handed over: a camera that has kept moving the same way for ``limit_s`` is a command left
        running (a joystick or an auto-tracker does not drive one way for a minute): stop it. True if it did."""
        last = self._expected
        dpan, dtilt = angle_delta(last.pan, where.pan), where.tilt - last.tilt
        if abs(dpan) < 1.0 and abs(dtilt) < 1.0:
            self._steady = None
            return False
        way = (math.copysign(1.0, dpan) if abs(dpan) >= 1.0 else 0.0, math.copysign(1.0, dtilt) if abs(dtilt) >= 1.0 else 0.0)
        if self._steady is None or self._steady[1] != way:
            self._steady = (now, way)
            return False
        if now - self._steady[0] < limit_s:
            return False
        self._steady = None
        with self._ptz_lock:
            self.ptz.continuous(0.0, 0.0, 0.0)
            self._last_command = now
        print(f"{time.strftime('%H:%M:%S')} hunter {self.config.camera_id[:8]} stopped the camera: it kept moving the "
              f"same way for {limit_s:.0f} s while handed over (pan {where.pan:.1f} tilt {where.tilt:.1f})", flush=True)
        self._runaway, self._handed_over_until, self._expected = None, now, None
        self.status.error = self.warning
        return True

    def _still_ours(self, now: float, where: View) -> bool:
        """After a hand-over that left the hunter's command running: if the camera keeps moving exactly as that
        command drives it (two checks running), nobody replaced it and the hand-over was a false alarm: stop the motor
        and carry on. Without this, a false alarm (a zoom readback still settling, say) left Back PTZ panning in
        circles at the tilt end stop for 15 minutes, each move extending the hand-over. True if it stopped the motor."""
        since, before, speeds, matches = self._runaway
        if now - since < 1.0:
            return False
        verdict = obeys(speeds, before, where, now - since)
        if not verdict:  # still, or moving some other way: whoever moved it has it
            self._runaway = None
            return False
        if matches + 1 < 2:
            self._runaway = (now, where, speeds, matches + 1)
            return False
        self._runaway = None
        with self._ptz_lock:
            self.ptz.continuous(0.0, 0.0, 0.0)
            self._last_command = now
        print(f"{time.strftime('%H:%M:%S')} hunter {self.config.camera_id[:8]} stopped its own move: the camera kept "
              f"going as the hunter drove it, so nobody had taken over (pan {where.pan:.1f} tilt {where.tilt:.1f})",
              flush=True)
        self._handed_over_until, self._expected = now, None
        self.status.error = self.warning
        return True

    def _handed_over(self, now: float, where: View, unstopped: dict[str, float] | None = None, why: str = "") -> None:
        """Someone else is moving the camera: an operator, or the camera's own auto-tracking. They come first.
        ``unstopped``: the hunter's own command, which it did not stop (it never stops someone else's move); the
        watcher checks that it really was replaced (``_runaway``)."""
        if unstopped:
            self._runaway = (now, where, unstopped, 0)
        if now >= self._handed_over_until:  # a new hand-over, not one extended by further movement
            print(f"{time.strftime('%H:%M:%S')} hunter {self.config.camera_id[:8]} handed over: moved by someone else to "
                  f"pan {where.pan:.1f} tilt {where.tilt:.1f} fov {where.fov:.1f}{f' ({why})' if why else ''}", flush=True)
        self._handed_over_until = now + self.config.handback_s
        self._expected = where  # further movement extends the hand-over
        self.status.view = None
        log_line = f"handed over: camera moved by someone else to pan {where.pan:.1f} tilt {where.tilt:.1f}"
        if self.status.error != log_line:
            self.status.error = log_line

    def _keepalive_step(self, now: float) -> bool:
        """One check; True if a stop command was sent."""
        if now - self._last_command < self.config.keepalive_s or now < self._quiet_until or now < self._handed_over_until:
            return False
        if not self._ptz_lock.acquire(blocking=False):  # a move is under way: that counts as activity
            return False
        try:
            self.ptz.continuous(0.0, 0.0, 0.0)
            self._last_command = now
            return True
        except Exception:  # noqa: BLE001 - the hunter's own loop reports Server trouble
            return False
        finally:
            self._ptz_lock.release()

    def stop(self, wait: bool = True, return_home: bool = True) -> None:
        """Stop hunting. With ``return_home`` False the motor just stops where it is (a replacement hunter takes
        over, and two hunters must never drive the same camera)."""
        self._return_home = return_home
        self._stop.set()
        if not return_home and self.clock() >= self._handed_over_until:  # never stop an operator's or tracker's move
            try:
                self.ptz.continuous(0.0, 0.0, 0.0)
            except Exception:  # noqa: BLE001
                pass
        if wait and self._thread:
            self._thread.join(timeout=300)

    def look_at(self, view: View) -> None:
        """Go there next and watch it a while (a search result's "Look there"); the hunt carries on after."""
        self._requested = View.of(view.pan, clamp_tilt(self.config, view.tilt), max(self.config.min_fov, view.fov))

    def take_over_from(self, old: "Hunter") -> None:
        """A replacement hunter (settings saved) keeps its predecessor's hand-over: an operator who changes the
        settings while driving the camera keeps it."""
        self._handed_over_until, self._expected = old._handed_over_until, old._expected

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
        version = data.get("version") if isinstance(data, dict) else None
        for row in data.get("views", []) if version in (2, MEMORY_VERSION) else []:  # older: other prompts, re-survey
            stats = ViewStats.from_json(row["stats"])
            if version == 2 and stats.base_avoid >= 0.8 and not stats.extra_avoid:
                # Version 2 could carry an operator's avoid veto ("Is there a tree in view?", yes almost everywhere)
                # in base_avoid, which blocks the view for good: a blocked view is never visited, so never rescored.
                stats.scored, stats.avoid, stats.base_avoid = False, 0.0, 0.0
            if not stats.scored and stats.scene >= 0.0:
                stats.prior, stats.scene = 0.5, -1.0  # to be scored again: neutral until then, so it gets visited
            self.stats[View(row["pan"], row["tilt"], row["fov"])] = stats
        self.stats = merge_duplicates(self.stats)
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
                if any(s.visits and same_spot(view, v) for v, s in self.stats.items()):
                    continue  # surveyed in an earlier run (perhaps merged into a view a few degrees off)
                self.stats.setdefault(view, ViewStats())
                if not self._wait_while_handed_over():
                    break
                try:
                    self._visit(view, survey=True)
                except ExternalControl as taken:
                    self._handed_over(self.clock(), taken.where, taken.unstopped, taken.why)
                except (OSError, ValueError) as error:  # the Server is unreachable (restarting, say)
                    if not self._recover(error):
                        break
            self.status.state = "hunting"
            while not self._stop.is_set():
                if not self._wait_while_handed_over():
                    break
                hour = datetime.fromtimestamp(self.clock()).hour
                visit_started = self.clock()
                requested, self._requested = self._requested, None
                view = requested or choose_next(
                    self._candidates(), self.status.view, self.clock(), hour, magnets=self.magnets,
                    magnet_weight=self.config.magnet_weight, magnet_radius=self.config.magnet_radius,
                    since_magnet_s=self.clock() - self.last_magnet_visit, magnet_return_s=self.config.magnet_return_s,
                    recent=list(self.recent), bored_weight=self.config.bored_weight)
                try:
                    self._visit(view, survey=False)
                    if requested is not None and self._mode != "wide":
                        self._hold()  # asked for: watch it a while, whatever the boredom cycle says
                except ExternalControl as taken:
                    self._handed_over(self.clock(), taken.where, taken.unstopped, taken.why)
                    continue
                except (OSError, ValueError) as error:  # the Server is unreachable (restarting, say)
                    if not self._recover(error):
                        break
                watched = self.status.view or view
                self.recent.append((self.clock(), watched))
                self._spent(watched, self.clock() - visit_started)
        except Exception as error:  # noqa: BLE001 - report, then hand the camera back
            fatal, self.status.error = True, repr(error)
        finally:
            if self.home is not None and self._return_home and self.clock() >= self._handed_over_until:
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
        """The boredom cycle: ``wide_share`` of each ``cycle_s`` watching wide views; then the hunter gets bored and
        looks around close up, at zoomed spots (found by the helper questions, or by a tile that looked like a yes),
        briefly, before going back to wide. With no zoomed spot worth a look it stays wide."""
        now = self.clock()
        if self._mode_until is None:  # a cycle starts by watching
            self._mode_until = now + self.config.wide_share * self.config.cycle_s
        if now >= self._mode_until:
            if self._mode == "wide":
                self._mode, self._mode_until = "looking around", now + (1.0 - self.config.wide_share) * self.config.cycle_s
            else:
                self._mode, self._mode_until = "wide", now + self.config.wide_share * self.config.cycle_s
        want_wide = self._mode == "wide" or self.config.wide_share >= 1.0
        chosen = {v: s for v, s in self.stats.items() if self._is_wide(v) == want_wide}
        if not any(s.avoid < 0.8 and not hopeless(s) for s in chosen.values()):
            self._mode = "wide"
            return {v: s for v, s in self.stats.items() if self._is_wide(v)} or self.stats
        return chosen

    def _wait_while_handed_over(self) -> bool:
        """While someone else has the camera, send nothing and wait; then carry on from wherever they left it.
        False if stopped meanwhile."""
        if self.clock() >= self._handed_over_until:
            return True
        phase, began = self.status.state, self.clock()
        self.status.state = "handed over"
        while self.clock() < self._handed_over_until:
            if not self._pause(2.0):
                return False
        if self._mode_until is not None:  # the boredom cycle pauses while someone else has the camera
            self._mode_until += self.clock() - began
        self.status.state, self.status.error = phase, self.warning
        try:
            self._expected = self.ptz.position()
        except Exception:  # noqa: BLE001 - the next move reads it again
            self._expected = None
        self.status.view = None  # the next move starts from wherever they left it
        return True

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
            with self._ptz_lock:
                after, zoomed = self._move(before, target)
                self._expected = after
            self._last_command = time.time()
            if zoomed:
                self._quiet_until = time.time() + self.config.focus_s
                # The zoom reads back late: until it has settled, the watcher has nothing to compare with (with the
                # zoom-out still reading 15.7 degrees after focus_s, the watcher took the hunter's own zoom for an
                # operator's).
                self._expected = None
        finally:
            # Refocusing after a zoom blurs the picture: windows over it are camera motion too.
            self.move_log[-1] = (started, self.clock() + self.config.settle_s + (self.config.focus_s if zoomed else 0.0))
        if before.distance(after) <= 1.0:
            self.move_log.pop()  # already there
        else:
            self.status.moves += 1
        self.move_log = self.move_log[-50:]
        # Close enough is the same view; otherwise the spot it is at (an existing view there, else a new one), so what
        # is learned is not spread over copies a degree or two apart.
        actual = View.of(after.pan, after.tilt, after.fov)
        arrived = view if same_spot(actual, view) else next((v for v in self.stats if same_spot(actual, v)), actual)
        if arrived != self.status.view:
            self._arrived_at = self.clock()
        self.status.view = arrived
        if not self._pause(self.config.settle_s + (self.config.focus_s if zoomed else 0.0)):
            return None
        if zoomed:  # the zoom reads back late: where it settled is where the hunter left it
            with self._ptz_lock:
                self._expected = self._settled_zoom(target.fov)
        return arrived

    def _settled_zoom(self, fov: float, wait_s: float = 8.0) -> View:
        """The position once the zoom reads back as asked for, or has read the same for 3 s (the zoom may stop short),
        or after ``wait_s``."""
        now, same_since, waited = self.ptz.position(), 0.0, 0.0
        while abs(now.fov - fov) > max(2.0, 0.1 * fov) and waited < wait_s:
            self.sleep(1.0)
            waited += 1.0
            again = self.ptz.position()
            same_since = same_since + 1.0 if abs(again.fov - now.fov) < 0.5 else 0.0
            now = again
            if same_since >= 3.0:
                break
        return now

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
        # Compared with where the camera really stopped, not ``view``: an arrival is learned as the known view at that
        # spot, which can be a quarter of a field of view away (4.7 degrees close up), so every look there handed over.
        ref = self._expected or view
        if abs(angle_delta(now.pan, ref.pan)) <= 2.5 and abs(now.tilt - ref.tilt) <= 2.5 \
                and abs(now.fov - ref.fov) <= max(2.0, 0.1 * ref.fov):
            return True
        self._handed_over(self.clock(), now, why=f"snapshot check: expected pan {ref.pan:.1f} tilt {ref.tilt:.1f} "
                                                f"fov {ref.fov:.1f}")
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
        if self.embedder is not None:
            # Every look: the frame and its tiles into the map (about 0.45 s next to Clef), and the view's good-place
            # share, which can rule it out before Clef's scene questions.
            try:
                boxes9 = tile_boxes(3)
                frame, tile_vectors, thumb = self.embedder.look(jpeg, boxes9)
                stats.embed = self.embedder.share(frame)
                self._look_vectors = (jpeg, tile_vectors)
                if self.embed_map is not None:
                    self.embed_map.add_look(view, frame, tile_vectors, boxes9, thumb)
            except Exception:  # noqa: BLE001 - the embedding server is down: Clef scores the view as before
                self._look_vectors = None
        if hopeless(stats):
            # Nothing a person could be on: Clef's scene questions would only confirm it. The hunt questions are
            # still asked (this look is not wasted), and the view is never chosen for hunting again.
            stats.scored, stats.scene = True, max(0.0, stats.embed)
        elif survey or not stats.scored or (cfg.cues and not stats.cues):
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
        boxes = tile_boxes(cfg.tiles) if stats.avoid < 0.8 and not hopeless(stats) else []
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
            if self._mode == "wide":  # looking around is a quick look at each spot; watching is holding
                self._hold()
        if not survey and stats.prior >= 0.6:  # explore around promising views
            half = view.fov * cfg.overlap / 2
            for pan in (view.pan - half, view.pan + half):
                neighbour = View.of(pan, view.tilt, view.fov)
                if not any(same_spot(neighbour, v) for v in self.stats):
                    self.stats[neighbour] = ViewStats(prior=stats.prior * 0.9)

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
        if self.embedder is not None:
            try:
                if self._look_vectors is None or self._look_vectors[0] is not jpeg or cfg.tiles != 3:
                    _, vectors, _ = self.embedder.look(jpeg, boxes)
                else:
                    vectors = self._look_vectors[1]  # this look's tiles were embedded already
                shares, sims = self.embedder.tile_scores(vectors, list(cfg.cues))
            except Exception:  # noqa: BLE001 - the embedding server is down: ask Clef tile by tile, as before
                shares = None
            if shares is not None:
                stats.cue_scan_at = self.clock()
                for index, score, cue in embedding.standout_tiles(sims)[:ZOOM_CHILDREN]:
                    if shares[index] < embedding.HOPELESS:
                        continue  # the cue stands out, but the tile is mostly wall, roof or leaves
                    child = zoom_on(view, boxes[index], cfg.min_fov)
                    if any(v.fov < view.fov and v.distance(child) < 0.25 * child.fov for v in self.stats):
                        continue
                    self.stats[child] = ViewStats(prior=stats.prior * (0.5 + 0.5 * score), found_by=cue,
                                                  embed=shares[index])
                self._save()
                return
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
        while not self._stop.is_set() and self.clock() >= self._handed_over_until:
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
