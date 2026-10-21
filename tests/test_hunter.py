import json
import math
import time

import pytest

from clef_operator.hunter import (
    CUE_RESCAN_S, DEFAULT_CUES, ExternalControl, camera_medians, hopeless, merge_duplicates, obeys, same_spot, FOV_RATE, MEMORY_VERSION, PAN_RATE, cue_boost, standout_tiles, zoom_on, QUIET_RECHECK_S, is_quiet, resting, sightings, TILT_RATE, TRACK_HOLD_S, ZOOM_TIME, zoom_seconds, DEFAULT_AVOID, Hunter, HunterConfig, View, ViewStats, aim_at,
    angle_delta, choose_next, likelihood_questions, move_slowly, prior_from, survey_views, tile_boxes,
)


class FakePtz:
    """Logical-space camera: moves at the commanded continuous speed while simulated time passes. ``lag`` makes the
    reported position refresh only that often, and ``sticky`` ignores a command that zeroes one axis while others
    move (the axis keeps going), as Back PTZ sometimes does."""

    def __init__(self, view=View(184.0, -12.0, 36.9), tilt_max=90.0, lag=0.0, sticky=False):
        self.pan, self.tilt, self.fov = view.pan, view.tilt, view.fov
        self.tilt_max, self.lag, self.sticky = tilt_max, lag, sticky
        self.speed = (0.0, 0.0, 0.0)
        self.max_speed_used = 0.0
        self.now = 0.0
        self.absolute_moves = []
        self.commands = []  # (time, "continuous" | "absolute") for every command sent
        self._seen, self._seen_at = None, -1e9

    def position(self):
        if self._seen is None or self.now - self._seen_at >= self.lag:
            self._seen, self._seen_at = View(self.pan % 360.0, self.tilt, self.fov), self.now
        return self._seen

    def continuous(self, pan, tilt, zoom):
        self.commands.append((self.now, "continuous"))
        new = (pan, tilt, zoom)
        if self.sticky and any(new):
            new = tuple(n if n or not o else o for n, o in zip(new, self.speed))
        self.speed = new
        self.max_speed_used = max(self.max_speed_used, abs(pan), abs(tilt))  # the motors: zoom is silent

    def absolute(self, view):
        self.commands.append((self.now, "absolute"))
        self.absolute_moves.append(view)
        self.pan, self.tilt, self.fov = view.pan, min(view.tilt, self.tilt_max), view.fov
        self._seen = None

    def snapshot(self):
        return b"jpeg"

    def sleep(self, seconds):
        self.pan += self.speed[0] * PAN_RATE * seconds
        self.tilt += self.speed[1] * TILT_RATE * seconds
        if self.tilt > self.tilt_max:  # like Back PTZ: driven up into the end stop, it pans by itself
            self.tilt = self.tilt_max
            self.pan -= 6.0 * seconds
        if self.speed[2]:  # like Back PTZ: one zoom rate whatever the speed, slower zoomed in (see zoom_seconds)
            travel = math.copysign(seconds / ZOOM_TIME, -self.speed[2])  # zoom in narrows the field of view
            self.fov = max(1.0, self.fov ** 0.6 + travel) ** (1 / 0.6)
        self.now += seconds


def no_crop(jpeg, x, y, w, h):
    return f"tile {x:.2f},{y:.2f}".encode()


def test_move_slowly_caps_speed_wraps_pan_and_stops_on_target():
    ptz = FakePtz(View(350.0, -12.0, 36.9))
    end = move_slowly(ptz, View(10.0, -5.0, 20.0), max_speed=0.1, sleep=ptz.sleep)
    assert abs(angle_delta(end.pan, 10.0)) <= 2.0 and abs(end.tilt + 5.0) <= 2.0 and abs(end.fov - 20.0) <= 2.0
    assert ptz.max_speed_used <= 0.1 + 1e-9 and ptz.speed == (0.0, 0.0, 0.0)
    assert ptz.now > 6  # 20 degrees at about 3 degrees a second: slow and quiet


def test_survey_sweeps_high_rows_first_and_only_visits_a_low_preset_nearby():
    cfg = HunterConfig("cam", pan_min=150, pan_max=250)
    home = View(184.0, -12.0, 36.9)
    views = survey_views(cfg, home, [View(248.0, -41.5, 36.9)])
    assert views[0].tilt == 5.0  # starts above home, not at the ground
    rows = [v.tilt for v in views]
    assert rows.count(5.0) == rows.count(-10.0) == rows.count(-30.0) == 4  # above, at and below home
    assert rows.count(-40.0) == 2  # low row: only either side of the preset, never a near copy of it
    assert all(10 <= abs(angle_delta(v.pan, 248.0)) <= 30 for v in views if v.tilt == -40.0)


def test_survey_rows_stay_clear_of_the_tilt_end_stop():
    cfg = HunterConfig("cam", pan_min=150, pan_max=250, tilt_max=5.0)
    rows = {v.tilt for v in survey_views(cfg, View(184.0, -12.0, 36.9))}
    assert max(rows) == 3.0 and -10.0 in rows


def test_negative_prompts_cover_walls_and_structures_below():
    texts = [q["instructions"] for k, q in likelihood_questions(["person"], avoid=["Is this the neighbour's yard?"]).items()
             if k.startswith("_avoid")]
    assert texts[: len(DEFAULT_AVOID)] == DEFAULT_AVOID and texts[-1] == "Is this the neighbour's yard?"
    assert any("wall" in t for t in texts) and any("pergola" in t for t in texts)


def kind(**p):
    return {k: p.get(k, 0.0) for k in ("ground", "plants", "building", "below", "sky")}


def test_a_lawn_framed_by_trees_beats_a_wall_and_a_pergola_below():
    # Answers Clef gave on Back PTZ (pan 160.5 tilt 0, pan 72 tilt -30, pan 160.5 tilt -30)
    lawn = {"_ground": {"true": 0.97}, "_foliage": {"true": 0.20}, "_kind": kind(ground=0.13, plants=0.85),
            "_avoid0": {"true": 0.00}, "_avoid1": {"true": 0.02}, "_avoid2": {"true": 0.04}}
    wall = {"_ground": {"true": 0.31}, "_foliage": {"true": 0.02}, "_kind": kind(building=0.95),
            "_avoid0": {"true": 0.61}, "_avoid1": {"true": 0.08}, "_avoid2": {"true": 0.00}}
    pergola = {"_ground": {"true": 0.19}, "_foliage": {"true": 0.09}, "_kind": kind(below=0.9),
               "_avoid0": {"true": 0.29}, "_avoid1": {"true": 0.85}, "_avoid2": {"true": 0.00}}
    (lawn_p, lawn_avoid), (wall_p, wall_avoid), (pergola_p, pergola_avoid) = map(prior_from, (lawn, wall, pergola))
    assert lawn_p > 0.7 and lawn_avoid < 0.1  # the lawn is not "vegetation to avoid"
    assert wall_p < 0.05 and wall_avoid >= 0.8 and pergola_p < 0.05 and pergola_avoid >= 0.8


def test_move_arrives_when_positions_come_late_and_one_axis_ignores_a_zero():
    # Back PTZ: positions refresh once a second, and "pan 0, tilt 0.03" can leave it panning at full speed
    ptz = FakePtz(View(173.8, -9.65, 36.9), lag=1.0, sticky=True)
    end = move_slowly(ptz, View(72.0, 3.0, 36.9), max_speed=0.2, sleep=ptz.sleep)
    assert abs(angle_delta(end.pan, 72.0)) <= 2.0 and abs(end.tilt - 3.0) <= 2.0 and ptz.speed == (0.0, 0.0, 0.0)
    assert not ptz.absolute_moves  # never a jump: absolute moves ignore their speed and are loud
    assert ptz.now < 120


def test_move_stops_at_the_tilt_end_stop_and_undoes_the_pan_it_caused():
    ptz = FakePtz(View(160.0, -25.0, 36.9), tilt_max=5.0)
    end = move_slowly(ptz, View(160.0, 10.0, 36.9), max_speed=0.2, sleep=ptz.sleep)
    assert end.tilt == 5.0 and ptz.speed == (0.0, 0.0, 0.0)
    assert abs(angle_delta(end.pan, 160.0)) <= 2.0  # pan drifted while tilt was stuck, then was put back
    assert ptz.now < 60  # not driving into the stop until the timeout


def test_views_are_learned_where_the_camera_really_is():
    ptz = FakePtz(View(184.0, -12.0, 36.9), tilt_max=5.0)
    cfg = HunterConfig("cam", pan_min=184, pan_max=184.5, tilts=[30.0], rest_s=1, settle_s=1, tiles=1)

    def ask(camera_id, jpeg, questions):
        hunter.stop(wait=False)
        return {q: {"true": 0.1} for q in questions}

    hunter = Hunter(cfg, ptz, ask, sleep=ptz.sleep, clock=lambda: ptz.now)
    hunter.run()
    assert all(v.tilt <= 5.0 for v, s in hunter.stats.items() if s.visits)  # never "visited" tilt 30


def test_answers_are_dropped_when_something_else_moved_the_camera():
    # The Server sends a camera with a PTZ home position home after a few idle minutes, e.g. while Clef is busy
    ptz = FakePtz(View(131.0, 3.0, 36.9))
    cfg = HunterConfig("cam", pan_min=131, pan_max=131.5, tilts=[3.0], rest_s=1, settle_s=1, tiles=1)

    def ask(camera_id, jpeg, questions):
        hunter.stop(wait=False)
        return {q: {"true": 0.9} for q in questions if q != "_kind"} | {"_kind": kind(ground=1.0)}

    def snapshot():
        ptz.pan, ptz.tilt = 183.9, -12.0  # sent home after the move, before the snapshot
        ptz._seen = None
        hunter.stop(wait=False)
        return b"jpeg"

    ptz.snapshot = snapshot
    hunter = Hunter(cfg, ptz, ask, sleep=ptz.sleep, clock=lambda: ptz.now)
    hunter.run()
    learned = hunter.stats[View.of(131.0, 3.0, 36.9)]
    assert learned.prior == 0.7 and not learned.total_hits  # home magnet: its starting prior, untouched
    assert "moved" in hunter.status.error


def test_answers_about_a_snapshot_stay_valid_however_long_clef_takes():
    ptz = FakePtz(View(131.0, 3.0, 36.9))
    cfg = HunterConfig("cam", pan_min=131, pan_max=131.5, tilts=[3.0], rest_s=1, settle_s=1, tiles=1)

    def ask(camera_id, jpeg, questions):
        ptz.pan, ptz.tilt = 183.9, -12.0  # sent home while the question waited in Clef's queue
        ptz._seen = None
        hunter.stop(wait=False)
        return {q: {"true": 0.9 if q == "_ground" else 0.05} for q in questions if q != "_kind"} | {"_kind": kind(ground=1.0)}

    hunter = Hunter(cfg, ptz, ask, sleep=ptz.sleep, clock=lambda: ptz.now)
    hunter.run()
    assert hunter.stats[View.of(131.0, 3.0, 36.9)].scene > 0.75  # learned from the snapshot taken there


def test_a_view_clef_never_scored_is_scored_on_the_next_visit():
    ptz = FakePtz(View(131.0, 3.0, 36.9))
    cfg = HunterConfig("cam", pan_min=131, pan_max=131.5, tilts=[3.0], rest_s=1, settle_s=1, tiles=1)
    asked = []

    def ask(camera_id, jpeg, questions):
        asked.append(set(questions))
        if len(asked) == 1:
            raise RuntimeError("hunter question not answered: dropped")  # Clef too busy
        hunter.stop(wait=False)
        return {q: {"true": 0.9 if q == "_ground" else 0.05} for q in questions if q != "_kind"} | {"_kind": kind(ground=1.0)}

    hunter = Hunter(cfg, ptz, ask, sleep=ptz.sleep, clock=lambda: ptz.now)
    hunter.run()
    assert "_ground" in asked[1]  # asked again, though the view was already visited
    assert hunter.stats[View.of(131.0, 3.0, 36.9)].scored


def test_a_good_view_across_the_yard_beats_bouncing_between_two_neighbours():
    here, neighbour, lawn = View(249.0, -30.0, 37.0), View(248.5, -41.5, 37.0), View(190.0, -10.0, 37.0)
    stats = {here: ViewStats(prior=0.57, visits=3, last_visit=2990.0),
             neighbour: ViewStats(prior=0.54, visits=3, last_visit=2960.0),
             lawn: ViewStats(prior=0.56, visits=1, last_visit=1000.0)}  # due a look: over half an hour ago
    assert choose_next(stats, here, now=3000.0, hour=9, magnets=[neighbour], since_magnet_s=40.0) == lawn


def test_magnets_pull_harder_the_longer_the_hunter_is_away():
    here, magnet = View(240.0, -10.0, 36.9), View(184.0, -10.0, 36.9)
    stats = {View(260.0, -10.0, 36.9): ViewStats(prior=0.6), View(200.0, -10.0, 36.9): ViewStats(prior=0.5),
             magnet: ViewStats(prior=0.5), here: ViewStats(prior=0.6, visits=1, last_visit=100.0)}
    kw = dict(now=200.0, hour=12, magnets=[magnet], magnet_weight=1.0, magnet_radius=25.0, magnet_return_s=600.0)
    assert choose_next(stats, here, since_magnet_s=0.0, **kw) == View(260.0, -10.0, 36.9)
    assert choose_next(stats, here, since_magnet_s=1200.0, **kw) in (magnet, View(200.0, -10.0, 36.9))


def test_aim_at_a_top_right_tile_pans_right_tilts_up_and_zooms_in():
    view = View(184.0, -12.0, 36.9)
    box = tile_boxes(3)[2]  # top right
    target = aim_at(view, box, min_fov=6.0)
    assert target.pan > view.pan and target.tilt > view.tilt and target.fov < view.fov


def test_hunter_finds_a_small_person_in_a_tile_and_zooms_in():
    ptz = FakePtz()
    found = []
    person = (190.0, -9.0)  # just right of and above the home view's centre: small in the full frame

    def tile_p(jpeg):
        x, y = (float(v) for v in jpeg.decode().split()[1].split(","))
        return 0.55 if (x > 0.4 and y < 0.3) else 0.05  # the person is in the top-right tile

    def ask(camera_id, jpeg, questions):
        if isinstance(jpeg, list):
            p = max(tile_p(j) for j in jpeg)  # all tiles at once: someone is in one of them
        elif jpeg.startswith(b"tile"):
            p = tile_p(jpeg)
        else:
            close = abs(angle_delta(ptz.pan, person[0])) < ptz.fov / 4 and abs(ptz.tilt - person[1]) < ptz.fov / 4
            p = 0.9 if close and ptz.fov < 30 else 0.1  # visible to Clef only when zoomed in
        if found:
            hunter.stop(wait=False)
        return {q: {"true": p if not q.startswith("_") else 0.3} for q in questions}

    cfg = HunterConfig("cam", pan_min=184, pan_max=184.5, tilts=[-12.0], rest_s=2, settle_s=1, dwell_s=10,
                       max_dwell_s=20, max_speed=0.1)
    hunter = Hunter(cfg, ptz, ask, on_found=lambda c, v, q, p: found.append((v, p)), crop=no_crop,
                    sleep=ptz.sleep, clock=lambda: 1_792_000_000 + ptz.now)
    hunter.run()
    assert found, "should have zoomed in on the tile and confirmed"
    view, p = found[0]
    assert view.fov < 36.9 and view.pan > 184.0 and p >= 0.7
    assert ptz.max_speed_used <= 0.1 + 1e-9
    assert hunter.status.state == "stopped" and abs(angle_delta(ptz.pan, 184.0)) <= 2.0  # back home


def test_an_empty_view_asks_all_tiles_in_one_request():
    ptz = FakePtz()
    asked = []

    def ask(camera_id, jpeg, questions):
        asked.append(len(jpeg) if isinstance(jpeg, list) else 1)
        if len(asked) >= 2:
            hunter.stop(wait=False)
        # Open ground nobody is on (all-low scene answers would make the view hopeless, and its tiles are not asked)
        return {q: {"true": 0.9 if q == "_ground" else 0.02} for q in questions if q != "_kind"} | {"_kind": kind(ground=1.0)}

    cfg = HunterConfig("cam", pan_min=184, pan_max=184.5, tilts=[-12.0], rest_s=1, settle_s=1)
    hunter = Hunter(cfg, ptz, ask, crop=no_crop, sleep=ptz.sleep, clock=lambda: ptz.now)
    hunter.run()
    assert asked == [1, 9]  # the frame, then the nine tiles together; no tile asked alone


def test_presets_become_magnets_and_bad_magnets_are_reported():
    ptz = FakePtz()
    cfg = HunterConfig("cam", presets=["1", "2"], magnets=[{"pan": 1, "tilt": 200}], pan_min=184, pan_max=184.5,
                       tilts=[-12.0], rest_s=1, settle_s=1, tiles=1)
    located = {"1": View(248.3, -41.6, 36.9), "2": View(183.9, -12.0, 36.9)}

    def ask(camera_id, jpeg, questions):
        hunter.stop(wait=False)
        return {q: {"true": 0.1} for q in questions}

    hunter = Hunter(cfg, ptz, ask, locate_preset=located.get, sleep=ptz.sleep, clock=lambda: ptz.now)
    assert "1,200" in hunter.status.error
    hunter.run()
    assert View.of(248.3, -41.6, 36.9) in hunter.magnets and View.of(183.9, -12.0, 36.9) in hunter.magnets


def test_moves_are_logged_so_windows_during_them_can_be_skipped():
    ptz = FakePtz(View(184.0, -12.0, 36.9))
    cfg = HunterConfig("cam", pan_min=184, pan_max=215, overlap=0.8, tilts=[-12.0], rest_s=1, settle_s=1, tiles=1)
    calls = []

    def ask(camera_id, jpeg, questions):
        calls.append(ptz.now)
        if len(calls) >= 2:
            hunter.stop(wait=False)
        return {q: {"true": 0.1} for q in questions}

    hunter = Hunter(cfg, ptz, ask, sleep=ptz.sleep, clock=lambda: ptz.now)
    hunter.run()
    assert len(hunter.move_log) == 1  # the first view was where the camera already was
    start, end = hunter.move_log[0]
    assert end - start > 5 and hunter.moved_during(start - 30, start + 1)
    assert not hunter.moved_during(start - 60, start - 1)


def test_hot_spots_pull_and_come_back_within_minutes():
    here = View(200.0, -10.0, 36.9)
    hot = View(150.0, -5.0, 12.0)
    ordinary = View(230.0, -10.0, 36.9)
    stats = {here: ViewStats(prior=0.5, visits=3, last_visit=900.0),
             ordinary: ViewStats(prior=0.6, visits=2, last_visit=500.0),
             hot: ViewStats(prior=0.5, visits=6, hits={12: 5}, checks={12: 6}, last_visit=600.0)}
    assert choose_next(stats, here, now=1000.0, hour=12) == hot  # 5 finds there: go back
    stats[hot].last_visit = 990.0
    assert choose_next(stats, here, now=1000.0, hour=12) == ordinary  # just checked: look elsewhere for now


def test_memory_survives_restarts(tmp_path):
    memory = tmp_path / "hunter.json"
    ptz = FakePtz()
    cfg = HunterConfig("cam", pan_min=184, pan_max=184.5, tilts=[-12.0], rest_s=1, settle_s=1, tiles=1)
    calls = []

    def ask(camera_id, jpeg, questions):
        calls.append(1)
        hunter.stop(wait=False)
        return {q: {"true": 0.9 if q == "t0" else 0.3} for q in questions}

    hunter = Hunter(cfg, ptz, ask, sleep=ptz.sleep, clock=lambda: 1_792_000_000 + ptz.now, memory=memory)
    hunter.run()
    assert memory.exists() and hunter.hot_spots()
    again = Hunter(cfg, FakePtz(), ask, memory=memory)
    again._load()
    assert again.hot_spots()[0][1].total_hits == hunter.hot_spots()[0][1].total_hits


def test_memory_from_older_prompts_is_forgotten_but_home_is_kept(tmp_path):
    memory = tmp_path / "hunter.json"
    memory.write_text('{"home": {"pan": 183.9, "tilt": -12.0, "fov": 36.9}, "views": [{"pan": 101.5, "tilt": -30.0, '
                      '"fov": 37.0, "stats": {"prior": 0.34, "avoid": 0.2, "visits": 4, "hits": {}, "checks": {}, '
                      '"last_visit": 1.0, "last_yes": 0.0}}]}')
    hunter = Hunter(HunterConfig("cam"), FakePtz(), lambda *a: {}, memory=memory)
    assert hunter._load() == View(183.9, -12.0, 36.9) and not hunter.stats
    hunter._save()
    assert f'"version": {MEMORY_VERSION}' in memory.read_text()


def test_home_preset_wins_over_where_the_camera_was_left(tmp_path):
    ptz = FakePtz(View(248.4, -45.8, 36.9))  # an earlier run left it aimed low, near preset 1
    cfg = HunterConfig("cam", home_preset="2", pan_min=150, pan_max=200, rest_s=1, settle_s=1, tiles=1)
    located = {"2": View(183.9, -12.0, 36.9)}

    def ask(camera_id, jpeg, questions):
        hunter.stop(wait=False)
        return {q: {"true": 0.1} for q in questions}

    hunter = Hunter(cfg, ptz, ask, locate_preset=located.get, sleep=ptz.sleep, clock=lambda: ptz.now,
                    memory=tmp_path / "m.json")
    hunter.run()
    assert hunter.home == View(183.9, -12.0, 36.9)
    assert abs(angle_delta(ptz.pan, 183.9)) <= 2.0 and abs(ptz.tilt + 12.0) <= 2.0  # stopped: went to the real home
    # without the preset, the remembered home is used, not the current position
    again = Hunter(HunterConfig("cam", tiles=1), FakePtz(View(248.4, -45.8, 36.9)), ask, memory=tmp_path / "m.json")
    assert again._load() == View(183.9, -12.0, 36.9)


def test_replaced_hunter_stops_its_motor_without_driving_home():
    ptz = FakePtz()
    ptz.continuous(0.1, 0.0, 0.0)
    hunter = Hunter(HunterConfig("cam"), ptz, lambda *a: {})
    hunter.stop(wait=False, return_home=False)
    assert ptz.speed == (0.0, 0.0, 0.0) and hunter._return_home is False


def test_views_where_nothing_shows_up_become_quiet_and_are_skipped():
    lawn, hedge = View(190.0, -10.0, 37.0), View(278.5, -10.0, 37.0)
    stats = {lawn: ViewStats(prior=0.6, visits=20, last_visit=100.0),
             hedge: ViewStats(prior=0.6, visits=20, last_visit=100.0)}  # the scene looked as good at both
    for _ in range(20):
        stats[lawn].observe(18, 0.4)  # people often crossing the lawn in the evening
        stats[hedge].observe(18, 0.01)
    best = stats[lawn].yield_rate(18)
    assert stats[hedge].yield_rate(18) < 0.1 * best and is_quiet(stats[hedge], 18, best)
    assert sum(stats[hedge].seen.values()) == 0.0  # Clef's noise on an empty view is not a sighting
    assert not is_quiet(stats[lawn], 18, best)
    seen = sightings(stats)
    assert resting(stats[hedge], 300.0, 18, best, seen)  # quiet: skipped
    assert not resting(stats[hedge], 100.0 + QUIET_RECHECK_S + 1, 18, best, seen)  # hours later: rechecked
    assert not resting(stats[lawn], 300.0, 18, best, seen)


def test_evening_sightings_do_not_make_a_view_busy_in_the_morning():
    s = ViewStats(prior=0.5)
    for _ in range(30):
        s.observe(22, 0.9)
    assert s.yield_rate(22) > 0.6 and s.yield_rate(10) < 0.3


def test_the_cameras_own_windows_count_only_when_the_view_was_held_throughout():
    ptz = FakePtz()
    hunter = Hunter(HunterConfig("cam", settle_s=2), ptz, lambda *a: {}, sleep=ptz.sleep, clock=lambda: 1_792_000_000 + ptz.now)
    view = View.of(184.0, -12.0, 36.9)
    hunter.stats[view] = ViewStats(prior=0.5)
    hunter.status.view, hunter.status.state = view, "holding"
    hunter._arrived_at = 1_792_000_000 + 10.0
    ptz.now = 50.0
    hunter.observe_window(1_792_000_000 + 5.0, 1_792_000_000 + 35.0, 0.9)  # began before the camera got there
    assert hunter.stats[view].total_looks == 0
    hunter.observe_window(1_792_000_000 + 15.0, 1_792_000_000 + 45.0, 0.9)
    assert hunter.stats[view].total_looks == 1 and hunter._window_yes_until > hunter.clock()


def test_the_hunter_holds_a_view_for_one_of_the_cameras_own_windows_only_when_they_come():
    ptz = FakePtz()
    cfg = HunterConfig("cam", hold_s=52.0)
    hunter = Hunter(cfg, ptz, lambda *a: {}, sleep=ptz.sleep, clock=lambda: ptz.now)
    hunter._mode = "looking around"  # close up: holding is only for the camera's own windows
    hunter._arrived_at = ptz.now
    hunter._hold()
    assert ptz.now == 0.0  # no windows arriving: nothing to wait for
    hunter.window_arrived()
    hunter._hold()
    assert 50.0 <= ptz.now <= 53.0  # waited the full hold: no clean window was answered
    hunter._arrived_at = ptz.now
    hunter.window_arrived()
    hunter._clean_window_at = ptz.now + 0.5  # answered just after arriving
    start = ptz.now
    hunter._hold()
    assert ptz.now - start <= 1.0


def test_no_view_is_quiet_while_the_target_has_not_shown_up_anywhere():
    a, b = View(190.0, -10.0, 37.0), View(249.0, -30.0, 37.0)
    stats = {a: ViewStats(prior=0.6), b: ViewStats(prior=0.6)}
    for _ in range(30):
        stats[a].observe(9, 0.004)
        stats[b].observe(9, 0.04)  # noise differences, nobody in the yard
    best = max(s.yield_rate(9) for s in stats.values())
    assert not any(is_quiet(s, 9, best, sightings(stats)) for s in stats.values())


def test_poor_views_get_little_curiosity_and_no_hold():
    lawn, wall = View(190.0, -10.0, 37.0), View(72.0, -10.0, 37.0)
    stats = {lawn: ViewStats(prior=0.7, visits=10, last_visit=900.0),
             wall: ViewStats(prior=0.22, avoid=0.63, visits=0)}  # never visited: exploration alone used to win
    here = View(160.0, -10.0, 37.0)
    stats[here] = ViewStats(prior=0.5, visits=10, last_visit=990.0)
    assert choose_next(stats, here, now=1000.0, hour=9) != wall
    ptz = FakePtz()
    hunter = Hunter(HunterConfig("cam"), ptz, lambda *a: {}, sleep=ptz.sleep, clock=lambda: ptz.now)
    hunter.stats = stats
    assert hunter._worth_holding(lawn) and not hunter._worth_holding(wall)


# Helper answers Clef gave for the 9 tiles of Back PTZ at pan 190, tilt -10: a path across the bottom right, picnic
# tables right of centre and bottom right, lawn ("yard") everywhere.
PICNIC_TILES = [
    {"path": 0.07, "seat": 0.02, "yard": 0.97}, {"path": 0.35, "seat": 0.02, "yard": 0.96},
    {"path": 0.23, "seat": 0.29, "yard": 0.97}, {"path": 0.29, "seat": 0.0, "yard": 0.96},
    {"path": 0.33, "seat": 0.92, "yard": 0.97}, {"path": 0.11, "seat": 0.95, "yard": 0.98},
    {"path": 0.34, "seat": 0.02, "yard": 0.94}, {"path": 0.72, "seat": 0.9, "yard": 0.97},
    {"path": 0.75, "seat": 0.93, "yard": 0.97},
]


def test_tiles_count_only_where_a_helper_answer_stands_out():
    found = standout_tiles(PICNIC_TILES)
    assert {i for i, _, _ in found} == {4, 5, 7, 8}  # the picnic tables (and the path through them)
    assert all(cue != "yard" for _, _, cue in found)  # yes on every tile: tells nothing


def test_helper_answers_raise_a_view_only_where_they_stand_out_from_the_camera():
    medians = {"Is there a yard?": 0.85, "Is there a path or walkway?": 0.14}
    lawn_only = {"Is there a yard?": 0.95, "Is there a path or walkway?": 0.1}
    path = {"Is there a yard?": 0.95, "Is there a path or walkway?": 0.72}
    assert cue_boost(lawn_only, medians) < 0.65 and cue_boost(path, medians) == 1.0
    questions = likelihood_questions(["person"], cues=DEFAULT_CUES)
    assert sum(q.startswith("_cue") for q in questions) == len(DEFAULT_CUES)


def test_the_hunter_adds_zoomed_views_on_tiles_where_helpers_stand_out():
    ptz = FakePtz()
    cfg = HunterConfig("cam", cues=["path", "seat", "yard"], camera_height_m=7.4)  # high enough that these tiles are far
    by_tile = {}

    def ask(camera_id, jpeg, questions):
        tile = PICNIC_TILES[by_tile[jpeg]]
        return {q: {"true": tile[cfg.cues[int(q[1:])]]} for q in questions}

    def crop(jpeg, x, y, w, h):
        key = f"tile {x:.2f},{y:.2f}".encode()
        by_tile[key] = len(by_tile) % 9
        return key

    hunter = Hunter(cfg, ptz, ask, crop=crop, sleep=ptz.sleep, clock=lambda: 1_000_000.0 + ptz.now)
    wide = View.of(190.0, -10.0, 37.0)
    hunter.home = wide
    hunter.stats[wide] = ViewStats(prior=0.8)
    hunter._scan_tiles(wide, b"jpeg")
    zoomed = [(v, s) for v, s in hunter.stats.items() if s.found_by]
    assert 1 <= len(zoomed) <= 3 and all(12.0 <= v.fov <= 24.0 for v, _ in zoomed)  # zoomed for their distance
    picnic = {zoom_on(wide, tile_boxes(3)[i], cfg.min_fov) for i in (4, 5, 7, 8)}
    assert {v for v, _ in zoomed} <= picnic  # centred on picnic-area tiles: centre, right, below, below right
    assert all(s.found_by in ("seat", "path") for _, s in zoomed)
    before = len(hunter.stats)
    hunter._scan_tiles(wide, b"jpeg")
    assert len(hunter.stats) == before  # not again within CUE_RESCAN_S
    ptz.now += CUE_RESCAN_S + 1
    hunter._scan_tiles(wide, b"jpeg")
    assert len(hunter.stats) == before  # rescanned, but no duplicate zoomed views


def test_the_hunter_waits_out_a_server_restart_instead_of_stopping():
    ptz = FakePtz()
    down = {"left": 0}
    real_position = ptz.position

    def position():
        if down["left"] > 0:
            down["left"] -= 1
            raise ConnectionRefusedError(111, "Connection refused")
        return real_position()

    ptz.position = position
    asked = []

    def ask(camera_id, jpeg, questions):
        asked.append(1)
        if len(asked) == 1:
            down["left"] = 3  # the Server restarts after the first look
        if len(asked) >= 4:
            hunter.stop(wait=False)
        return {q: {"true": 0.1} for q in questions if q != "_kind"} | {"_kind": kind(ground=1.0)}

    cfg = HunterConfig("cam", pan_min=150, pan_max=220, tilts=[-12.0], rest_s=1, settle_s=1, tiles=1)
    hunter = Hunter(cfg, ptz, ask, sleep=ptz.sleep, clock=lambda: ptz.now)
    hunter.run()
    assert len(asked) >= 4 and hunter.status.state == "stopped"  # kept going after the outage


def test_an_avoid_question_that_is_yes_everywhere_does_not_rule_out_every_view():
    ptz = FakePtz()
    cfg = HunterConfig("cam", avoid=["is there a tree in view?", "is this siding of a house?"])
    hunter = Hunter(cfg, ptz, lambda *a: {})
    views = {View(190.0 + 30 * i, -10.0, 37.0): (0.97, 0.02) for i in range(4)}  # trees in every view of the yard
    views[View(72.0, -10.0, 37.0)] = (0.95, 0.93)  # the house wall
    for view, (tree, siding) in views.items():
        hunter.stats[view] = ViewStats(scene=0.6, base_avoid=0.05, scored=True,
                                       extra_avoid={cfg.avoid[0]: tree, cfg.avoid[1]: siding})
    hunter._reprice()
    yard, wall = hunter.stats[View(190.0, -10.0, 37.0)], hunter.stats[View(72.0, -10.0, 37.0)]
    assert yard.avoid < 0.8 and yard.prior > 0.3  # trees everywhere: no reason to avoid this one
    assert wall.avoid >= 0.8 and wall.prior < 0.1  # siding stands out here


def test_views_saved_before_avoid_weighting_are_scored_again():
    old = {"prior": 0.02, "avoid": 0.98, "visits": 3, "hits": {}, "checks": {}, "last_visit": 1.0, "last_yes": 0.0,
           "scored": True, "cues": {"Is there a yard?": 0.97}}
    s = ViewStats.from_json(old)
    assert not s.scored and s.avoid == 0.0  # visited (not vetoed) and asked again


def test_zooming_is_continuous_after_a_slow_pan_and_then_the_camera_refocuses():
    ptz = FakePtz(View(184.0, -12.0, 37.0))
    sent = []
    real = ptz.continuous
    def continuous(pan, tilt, zoom):
        sent.append((ptz.now, pan, tilt, zoom))
        real(pan, tilt, zoom)
    ptz.continuous = continuous
    hunter = Hunter(HunterConfig("cam", focus_s=4.0, settle_s=1.0, rest_s=0.0), ptz, lambda *a: {},
                    sleep=ptz.sleep, clock=lambda: ptz.now)
    arrived = hunter._go(View(200.0, -15.0, 18.5))
    assert abs(arrived.pan - 200.0) <= 2.0 and abs(arrived.fov - 18.5) <= 2.5
    assert not ptz.absolute_moves  # never an absolute move: the camera finished them on its own, zoomed in
    assert not any(z and (p or t) for _, p, t, z in sent)  # zoom and pan never together
    last_pan = max(t for t, p, ti, z in sent if p or ti)
    first_zoom = min(t for t, p, ti, z in sent if z)
    assert first_zoom >= last_pan  # centred first (slowly), then zoomed in
    sent.clear()
    hunter._go(View(150.0, -10.0, 37.0))
    assert min(t for t, p, ti, z in sent if z) <= min(t for t, p, ti, z in sent if p or ti)  # widened, then the pan


def test_no_commands_are_sent_while_the_camera_refocuses():
    ptz = FakePtz(View(184.0, -12.0, 37.0))
    hunter = Hunter(HunterConfig("cam", focus_s=4.0, settle_s=1.0, rest_s=0.0), ptz, lambda *a: {},
                    sleep=ptz.sleep, clock=lambda: ptz.now)
    hunter._go(View(184.0, -12.0, 18.5))  # zoom only
    zoom_at = ptz.commands[-1][0]
    hunter._go(View(190.0, -12.0, 18.5))
    later = [t for t, _ in ptz.commands if t > zoom_at]
    assert later and min(later) >= zoom_at + 4.0


def test_the_hunter_stays_wide_until_bored_of_its_best_wide_views_then_looks_around_briefly():
    now = [0.0]
    hunter = Hunter(HunterConfig("cam"), FakePtz(), lambda *a: {}, clock=lambda: now[0])
    hunter.home = View(184.0, -12.0, 37.0)
    wides = [View(150.0, -10.0, 37.0), View(190.0, -10.0, 37.0), View(240.0, -25.0, 37.0)]
    close = View(175.0, -5.0, 18.5)
    hunter.stats = {v: ViewStats(prior=0.6) for v in wides} | {close: ViewStats(prior=0.6, found_by="play area")}
    assert set(hunter._candidates()) == set(wides)  # wide is its stance
    now[0] = 400.0
    assert set(hunter._candidates()) == set(wides)  # wide a while, but its best views have not been watched lately
    hunter.recent.extend((390.0, v) for v in wides)  # now each has been
    assert set(hunter._candidates()) == {close} and hunter._mode == "looking around"
    for _ in range(3):  # a look-around is a few close-up spots
        hunter._closer_left -= 1
    assert set(hunter._candidates()) == set(wides) and hunter._mode == "wide"
    now[0] = 400.0 + 901.0
    hunter.recent.clear()
    assert set(hunter._candidates()) == {close}  # wide for 15 minutes: looks around anyway
    hunter._mode, hunter._wide_since = "wide", now[0] - 2000.0
    hunter.stats[close] = ViewStats(prior=0.6, avoid=0.9)
    assert set(hunter._candidates()) == set(wides)  # nothing close worth a look (blocked): stays wide


def test_a_tile_much_more_like_what_it_looks_for_is_zoomed_in_on_then_the_hunter_goes_wide():
    sims = [{"Is there a person in view?": 0.62} for _ in range(9)]
    sims[4] = {"Is there a person in view?": 0.75}  # the middle tile looks far more like a person than usual
    asked = []
    hunter, view = _looking_hunter(FakeEmbedder(share=0.5, tile_shares=[0.5] * 9, tile_sims=sims), asked)
    hunter.home = view
    hunter.config.camera_height_m = 40.0  # high up: the middle tile is far away
    for pan in (100.0, 150.0, 200.0, 250.0):  # the camera's usual views, for the median
        hunter.stats[View(pan, -10.0, 36.9)] = ViewStats(likeness={"Is there a person in view?": 0.62})
    hunter._visit(view, survey=False)
    zoomed = [v for v in hunter.stats if v.fov < 36.9]
    assert zoomed and abs(zoomed[0].pan - view.pan) < 3.0  # centred on the middle tile, zoomed in
    assert ["t"] in asked  # and Clef was asked about it there
    assert hunter.zoom_reason == ""  # done: the next move is wide again


def test_tour_close_ups_never_go_a_second_level_tighter_than_14_degrees():
    ptz = FakePtz()
    cfg = HunterConfig("cam", cues=["path", "seat", "yard"], camera_height_m=7.4)  # high enough that these tiles are far
    by_tile = {}

    def ask(camera_id, jpeg, questions):
        tile = PICNIC_TILES[by_tile[jpeg]]
        return {q: {"true": tile[cfg.cues[int(q[1:])]]} for q in questions}

    def crop(jpeg, x, y, w, h):
        key = f"tile {x:.2f},{y:.2f}".encode()
        by_tile[key] = len(by_tile) % 9
        return key

    hunter = Hunter(cfg, ptz, ask, crop=crop, sleep=ptz.sleep, clock=lambda: 1_000_000.0 + ptz.now)
    hunter.home = View.of(184.0, -12.0, 37.0)
    closer = View.of(175.0, -5.0, 18.5)
    hunter.stats[closer] = ViewStats(prior=0.8, found_by="play area")
    hunter._scan_tiles(closer, b"jpeg")
    # From an 18.5 degree close-up, the next level would be 9: tight zooms are for evidence, not touring
    assert not any(v.fov < 14.0 for v, s in hunter.stats.items() if s.found_by in ("seat", "path"))


def test_the_hunter_never_sends_an_absolute_move():
    ptz = FakePtz(View(184.0, -12.0, 37.0), lag=1.0)
    hunter = Hunter(HunterConfig("cam", rest_s=0.0, settle_s=1.0, focus_s=4.0), ptz, lambda *a: {},
                    sleep=ptz.sleep, clock=lambda: ptz.now)
    for view in (View(248.5, -41.5, 37.0), View(184.0, -12.0, 37.0), View(190.0, -10.0, 18.5), View(72.0, -10.0, 37.0)):
        hunter._go(view)
    assert ptz.absolute_moves == []


def test_keepalive_sends_a_stop_only_when_idle_and_never_while_refocusing_or_moving():
    ptz = FakePtz()
    hunter = Hunter(HunterConfig("cam", keepalive_s=60.0), ptz, lambda *a: {})
    hunter._last_command = 1000.0
    assert not hunter._keepalive_step(1030.0)  # a command 30 s ago: not idle yet
    hunter._quiet_until = 1070.0
    assert not hunter._keepalive_step(1065.0)  # idle, but the camera is refocusing after a zoom
    with hunter._ptz_lock:
        assert not hunter._keepalive_step(1080.0)  # a move holds the PTZ
    assert hunter._keepalive_step(1080.0) and ptz.commands[-1][1] == "continuous" and ptz.speed == (0.0, 0.0, 0.0)
    assert not hunter._keepalive_step(1100.0)  # just sent one


def test_views_blocked_by_an_old_avoid_veto_are_scored_again(tmp_path):
    memory = tmp_path / "hunter.json"
    stuck = {"prior": 0.02, "avoid": 0.98, "visits": 3, "hits": {}, "checks": {}, "last_visit": 1.0, "last_yes": 0.0,
             "scored": True, "base_avoid": 0.98, "extra_avoid": {}}
    wall = {**stuck, "extra_avoid": {"is this siding of a house?": 0.95}}  # vetoed by today's rules: stays blocked
    memory.write_text(json.dumps({"version": 2, "home": None, "views": [
        {"pan": 190.0, "tilt": -10.0, "fov": 37.0, "stats": stuck}, {"pan": 72.0, "tilt": -10.0, "fov": 37.0, "stats": wall}]}))
    hunter = Hunter(HunterConfig("cam"), FakePtz(), lambda *a: {}, memory=memory)
    hunter._load()
    lawn, house = hunter.stats[View(190.0, -10.0, 37.0)], hunter.stats[View(72.0, -10.0, 37.0)]
    assert not lawn.scored and lawn.avoid == 0.0 and lawn.prior == 0.5  # neutral, so it gets visited and rescored
    assert house.scored and house.avoid == 0.98


def test_near_duplicate_views_are_merged_with_their_history():
    a, b, c = View(194.5, -16.0, 37.0), View(190.5, -13.5, 37.0), View(249.0, -30.0, 37.0)  # a and b: 6.5 degrees apart
    stats = {a: ViewStats(prior=0.3, visits=50, looks={10: 40}, seen={10: 9.0}, hits={10: 2}),
             b: ViewStats(prior=0.5, visits=20, looks={10: 20, 11: 3}, seen={10: 1.0}),
             c: ViewStats(prior=0.6, visits=5)}
    merged = merge_duplicates(stats)
    assert set(merged) == {a, c} and same_spot(a, b) and not same_spot(a, View(194.5, -16.0, 18.5))
    assert merged[a].visits == 70 and merged[a].looks == {10: 60, 11: 3} and merged[a].seen[10] == 10.0
    assert merged[a].prior == 0.3  # the most visited copy keeps its scores


def test_a_move_that_ends_a_little_off_counts_as_the_view_it_was_for():
    ptz = FakePtz(View(184.0, -12.0, 37.0), lag=1.0)
    hunter = Hunter(HunterConfig("cam", rest_s=0.0, settle_s=1.0), ptz, lambda *a: {}, sleep=ptz.sleep, clock=lambda: ptz.now)
    target = View(220.0, -10.0, 37.0)
    hunter.stats[target] = ViewStats()
    assert hunter._go(target) == target and len(hunter.stats) == 1


def test_avoid_answers_lower_a_view_even_below_the_veto():
    here = View(100.0, -10.0, 37.0)
    cleaner, branches = View(190.0, -10.0, 37.0), View(194.0, -20.0, 37.0)
    stats = {here: ViewStats(prior=0.1, visits=9, last_visit=999.0),
             cleaner: ViewStats(prior=0.6, visits=3, last_visit=500.0, avoid=0.05),
             branches: ViewStats(prior=0.6, visits=3, last_visit=500.0, avoid=0.5)}
    assert choose_next(stats, here, now=1000.0, hour=9) == cleaner


def test_boredom_moves_the_hunter_on_from_an_area_it_has_been_watching():
    lawn_a, lawn_b, playground = View(186.0, -12.0, 37.0), View(194.0, -16.0, 37.0), View(249.0, -30.0, 37.0)
    stats = {lawn_a: ViewStats(prior=0.6, visits=40, last_visit=900.0), lawn_b: ViewStats(prior=0.6, visits=40, last_visit=900.0),
             playground: ViewStats(prior=0.55, visits=40, last_visit=900.0)}
    kw = dict(now=1000.0, hour=9, exploration=0.0)
    # lawn_b is 9 degrees from lawn_a: the same picture, so the hunter goes somewhere new, not back and forth
    assert choose_next(stats, lawn_a, **kw) == playground
    recent = [(1000.0 - 25 * i, playground) for i in range(12)]  # five minutes at the playground
    assert choose_next(stats, playground, recent=recent, **kw) in (lawn_a, lawn_b)



def test_a_hot_spot_near_home_loses_to_a_fresh_area_once_watched_again_and_again():
    hot, far = View(175.0, -10.0, 37.0), View(101.5, -10.0, 37.0)
    stats = {hot: ViewStats(prior=0.5, visits=40, last_visit=0.0, hits={10: 3}, looks={10: 40}, seen={10: 6.0}),
             far: ViewStats(prior=0.5, visits=40, last_visit=900.0, looks={10: 40}, seen={10: 4.0})}
    kw = dict(now=1000.0, hour=10, exploration=0.0, magnets=[View(184.0, -12.0, 37.0)], since_magnet_s=600.0)
    between = View(130.0, -10.0, 37.0)
    assert choose_next(stats, between, **kw) == hot  # finds there and the home magnet pull it back
    recent = [(1000.0 - 60 * i, hot) for i in range(4)]  # watched four times in the last few minutes
    assert choose_next(stats, between, recent=recent, **kw) == far



def test_boredom_never_sends_the_hunter_to_a_wall():
    lawn, house_side = View(175.0, -10.0, 37.0), View(131.0, -10.0, 37.0)
    stats = {lawn: ViewStats(prior=0.4, visits=40, last_visit=990.0, hits={10: 3}, looks={10: 40}, seen={10: 4.0},
                             scored=True, scene=0.43, base_avoid=0.43, avoid=0.43),
             house_side: ViewStats(prior=0.04, visits=2, last_visit=0.0, scored=True, scene=0.09, base_avoid=0.68,
                                   avoid=0.68)}
    recent = [(1000.0 - 30 * i, lawn) for i in range(20)]  # the lawn has been watched for ten minutes
    assert hopeless(stats[house_side]) and not hopeless(stats[lawn])
    assert choose_next(stats, lawn, now=1000.0, hour=10, recent=recent) == lawn

def test_a_bored_area_still_beats_a_quiet_view():
    lawn, quiet_home = View(175.0, -10.0, 37.0), View(186.0, -12.0, 37.0)
    stats = {quiet_home: ViewStats(prior=0.5, visits=500, last_visit=990.0, looks={10: 500}),
             lawn: ViewStats(prior=0.5, visits=40, last_visit=900.0, hits={10: 20}, looks={10: 40}, seen={10: 20.0})}
    recent = [(1000.0 - 20 * i, lawn) for i in range(10)]
    assert choose_next(stats, lawn, now=1000.0, hour=10, recent=recent) == lawn  # quiet stays skipped

def test_someone_else_moving_the_camera_hands_it_over_and_silences_the_hunter():
    ptz = FakePtz(View(190.0, -10.0, 37.0))
    hunter = Hunter(HunterConfig("cam", handback_s=180.0, keepalive_s=60.0), ptz, lambda *a: {})
    hunter._expected = View(190.0, -10.0, 37.0)
    hunter._last_command = 0.0
    assert not hunter._watch(1000.0)  # still where the hunter left it
    ptz.pan = 230.0  # an operator panned it (or the camera's auto-tracking followed someone)
    assert hunter._watch(1001.0) and hunter._handed_over_until == 1181.0
    assert not hunter._keepalive_step(1100.0)  # no stop command may interrupt them
    ptz.pan = 240.0
    assert hunter._watch(1150.0) and hunter._handed_over_until == 1330.0  # still moving: hand-back moves on
    assert not hunter._watch(1400.0)  # still since: the hunter takes over again



def test_settings_saved_while_handed_over_neither_stop_the_operator_nor_restart_the_hunt():
    ptz = FakePtz(View(190.0, -10.0, 37.0))
    old = Hunter(HunterConfig("cam"), ptz, lambda *a: {}, sleep=ptz.sleep, clock=lambda: ptz.now)
    old._handed_over(ptz.now, View(230.0, -10.0, 37.0))
    ptz.speed, sent = (0.3, 0.0, 0.0), len(ptz.commands)  # the operator is panning
    old.stop(wait=False, return_home=False)
    assert len(ptz.commands) == sent  # no stop on top of their move
    new = Hunter(HunterConfig("cam", wide_share=1.0), ptz, lambda *a: {}, sleep=ptz.sleep, clock=lambda: ptz.now)
    new.take_over_from(old)
    assert new._handed_over_until == old._handed_over_until and new._expected == View(230.0, -10.0, 37.0)


def test_a_false_hand_over_that_left_the_hunters_own_move_running_is_caught_and_stopped():
    # 11:31 on Back PTZ: a false hand-over mid-move left the hunter's pan-and-tilt command running (it never stops
    # someone else's move); tilt hit its end stop and the camera panned in circles for 15 minutes, every move
    # extending the hand-over.
    ptz = FakePtz(View(246.7, -42.7, 36.9), tilt_max=5.0)
    hunter = Hunter(HunterConfig("cam", handback_s=10.0), ptz, lambda *a: {}, sleep=ptz.sleep, clock=lambda: ptz.now)
    ptz.speed = (-0.2, 0.2, 0.0)  # the hunter's command, still in force
    hunter._handed_over(ptz.now, ptz.position(), {"pan": -0.2, "tilt": 0.2})
    for _ in range(4):
        ptz.sleep(2.0)
        hunter._watch(ptz.now)
    assert ptz.speed == (0.0, 0.0, 0.0) and ptz.now >= hunter._handed_over_until  # stopped, and the hunter carries on



def test_a_runaway_slowed_by_the_zoom_is_still_recognised_as_the_hunters():
    # 12:00 on Back PTZ: zoomed in to 17 degrees, speed 0.2 panned about 2 degrees a second, not 6.
    ptz = FakePtz(View(205.5, -16.8, 17.0))
    hunter = Hunter(HunterConfig("cam", handback_s=10.0), ptz, lambda *a: {}, sleep=ptz.sleep, clock=lambda: ptz.now)
    hunter._handed_over(ptz.now, ptz.position(), {"pan": 0.2, "tilt": -0.02})
    ptz.speed = (0.07, -0.007, 0.0)  # the same command, slowed by the zoom
    for _ in range(4):
        ptz.sleep(2.0)
        hunter._watch(ptz.now)
    assert ptz.speed == (0.0, 0.0, 0.0)


def test_a_camera_that_keeps_moving_one_way_for_a_minute_while_handed_over_is_stopped():
    ptz = FakePtz(View(200.0, -10.0, 36.9))
    hunter = Hunter(HunterConfig("cam", handback_s=10.0), ptz, lambda *a: {}, sleep=ptz.sleep, clock=lambda: ptz.now)
    hunter._handed_over(ptz.now, ptz.position())  # no command of the hunter's known to be running
    ptz.speed = (0.05, 0.0, 0.0)
    for _ in range(20):
        ptz.sleep(2.0)
        hunter._watch(ptz.now)
    assert ptz.speed != (0.0, 0.0, 0.0)  # 40 s: could still be someone's
    for _ in range(12):
        ptz.sleep(2.0)
        hunter._watch(ptz.now)
    assert ptz.speed == (0.0, 0.0, 0.0)

def test_a_real_operator_after_a_hand_over_is_left_alone():
    ptz = FakePtz(View(246.7, -20.0, 36.9))
    hunter = Hunter(HunterConfig("cam", handback_s=10.0), ptz, lambda *a: {}, sleep=ptz.sleep, clock=lambda: ptz.now)
    hunter._handed_over(ptz.now, ptz.position(), {"pan": -0.2, "tilt": 0.2})
    ptz.speed = (0.4, 0.0, 0.0)  # their joystick replaced the hunter's command
    sent = len(ptz.commands)
    for _ in range(4):
        ptz.sleep(2.0)
        hunter._watch(ptz.now)
    assert len(ptz.commands) == sent and ptz.now < hunter._handed_over_until


def test_a_zoom_still_settling_during_a_pan_is_not_an_operator():
    ptz = FakePtz(View(200.0, -10.0, 28.6))
    real_sleep = ptz.sleep

    def sleep(seconds):
        real_sleep(seconds)
        if ptz.fov < 36.9:
            ptz.fov = min(36.9, ptz.fov + 3.0 * seconds)  # the zoom-out finishing while the hunter pans
    ptz.sleep = sleep
    where = move_slowly(ptz, View(230.0, -10.0, 36.9), max_speed=0.2, sleep=ptz.sleep)
    assert abs(where.pan - 230.0) <= 2.5 and ptz.speed == (0.0, 0.0, 0.0)



def test_a_look_at_a_view_learned_a_few_degrees_off_is_not_a_hand_over():
    ptz = FakePtz(View(238.2, -22.7, 18.8))
    hunter = Hunter(HunterConfig("cam"), ptz, lambda *a: {}, sleep=ptz.sleep, clock=lambda: ptz.now)
    hunter._expected = ptz.position()  # where the move ended
    assert hunter._still_at(View(242.0, -24.0, 18.8))  # the known view at that spot, 4 degrees away
    ptz.pan = 250.0
    assert not hunter._still_at(View(242.0, -24.0, 18.8)) and hunter._handed_over_until > ptz.now


def test_after_a_zoom_the_hunter_waits_for_the_zoom_to_read_back_before_watching():
    ptz = FakePtz(View(185.8, -13.7, 15.7), lag=1.0)  # Back PTZ refreshes its position about once a second
    ptz.position()
    hunter = Hunter(HunterConfig("cam", settle_s=0.0, focus_s=0.0), ptz, lambda *a: {}, sleep=ptz.sleep,
                    clock=lambda: ptz.now)
    hunter.home = View(185.8, -13.7, 36.9)
    hunter._go(View(185.8, -13.7, 36.9))
    assert abs(hunter._expected.fov - 36.9) < 2.5  # where the zoom settled
    assert not hunter._watch(ptz.now)


def test_a_close_up_look_is_learned_at_the_close_up_even_when_the_zoom_reads_back_late():
    class LateZoom(FakePtz):  # 17:13 on Back PTZ: at 9 degrees, still reading 37 a second after zooming in
        def absolute(self, view):
            seen, at = self._seen, self._seen_at
            super().absolute(view)
            self._seen, self._seen_at = seen, at
    ptz = LateZoom(View(194.5, -16.0, 36.9), lag=2.5)
    ptz.position()
    hunter = Hunter(HunterConfig("cam", settle_s=0.0, focus_s=0.0), ptz, lambda *a: {}, sleep=ptz.sleep,
                    clock=lambda: ptz.now)
    hunter.home = View(184.0, -12.0, 36.9)
    wide, close = View(194.5, -16.0, 37.0), View(194.5, -16.0, 9.0)
    hunter.stats = {wide: ViewStats(visits=10), close: ViewStats()}
    assert hunter._go(close) == close and hunter.status.view == close

def test_obeys_tells_the_hunters_command_from_someone_elses():
    a = View(200.0, -10.0, 36.9)
    assert obeys({"pan": 0.2}, a, View(212.0, -10.0, 36.9), 2.0)  # 6 degrees a second, as commanded
    assert obeys({"pan": -0.2, "tilt": 0.2}, View(200.0, 5.0, 36.9), View(188.0, 5.0, 36.9), 2.0)  # tilt at its stop
    assert obeys({"pan": 0.2}, a, View(188.0, -10.0, 36.9), 2.0) is False  # the other way
    assert obeys({"pan": 0.2}, a, View(200.0, -10.0, 36.9), 2.0) is None  # standing still

def test_a_move_meeting_someone_elses_control_stops_without_sending_a_stop():
    ptz = FakePtz(View(190.0, -10.0, 37.0))
    real_sleep = ptz.sleep

    def sleep(seconds):
        real_sleep(seconds)
        if ptz.now > 1.0:
            ptz.pan -= 2.0  # someone else panning while the hunter only tilts
    ptz.sleep = sleep
    with pytest.raises(ExternalControl):
        move_slowly(ptz, View(190.0, -20.0, 37.0), max_speed=0.2, sleep=ptz.sleep)
    assert ptz.speed != (0.0, 0.0, 0.0)  # the hunter's last command was not a stop on top of theirs


def test_the_hunter_waits_while_handed_over_then_carries_on_from_where_it_was_left():
    ptz = FakePtz()
    hunter = Hunter(HunterConfig("cam"), ptz, lambda *a: {}, sleep=ptz.sleep, clock=lambda: ptz.now)
    hunter._handed_over_until = 60.0
    sent = len(ptz.commands)
    assert hunter._wait_while_handed_over() and ptz.now >= 60.0
    assert len(ptz.commands) == sent and hunter.status.view is None and hunter._expected is not None



def test_the_boredom_cycle_pauses_while_handed_over():
    ptz = FakePtz()
    hunter = Hunter(HunterConfig("cam"), ptz, lambda *a: {}, sleep=ptz.sleep, clock=lambda: ptz.now)
    hunter._mode, hunter._mode_until = "looking around", ptz.now + 50.0
    hunter._handed_over_until = ptz.now + 180.0
    assert hunter._wait_while_handed_over()
    assert hunter._mode == "looking around" and hunter._mode_until - ptz.now >= 49.0  # its close-up time is kept

def test_the_survey_skips_spots_already_known_under_a_merged_view():
    ptz = FakePtz(View(186.0, -12.0, 37.0))
    asked = []

    def ask(camera_id, jpeg, questions):
        asked.append(1)
        hunter.stop(wait=False)
        return {q: {"true": 0.1} for q in questions if q != "_kind"} | {"_kind": kind(ground=1.0)}

    cfg = HunterConfig("cam", pan_min=190, pan_max=190.5, tilts=[-10.0], rest_s=0, settle_s=0, tiles=1)
    hunter = Hunter(cfg, ptz, ask, sleep=ptz.sleep, clock=lambda: ptz.now)
    hunter.stats[View(186.0, -12.0, 37.0)] = ViewStats(prior=0.5, visits=40, scored=True)
    hunter.home = View(186.0, -12.0, 37.0)
    hunter.run()
    assert View.of(190.0, -10.0, 37.0) not in hunter.stats  # 190/-10 is 186/-12's spot: not surveyed again


class FakeEmbedder:
    """Stands in for the embedding server: a fixed good-place share per snapshot, and per-tile shares and cue
    similarities for tile scans. ``down`` makes every call fail, as when the server is not running."""

    def __init__(self, share=0.5, tile_shares=None, tile_sims=None, down=False):
        self.share_value, self.tile_shares, self.tile_sims, self.down = share, tile_shares, tile_sims, down
        self.calls = []

    def look(self, jpeg, boxes, thumb=320):
        self.calls.append("look")
        if self.down:
            raise ConnectionError("embedding server down")
        return [1.0, 0.0], [[0.0, 1.0] for _ in boxes], b"thumb"

    def share(self, vector):
        return self.share_value

    def likeness(self, frame, tiles, wanted, avoid):
        return self.share_value, {d: 0.65 for d in wanted}, {d: 0.6 for d in avoid}

    def tile_scores(self, vectors, cues):
        return self.tile_shares, self.tile_sims

    def content_model(self, examples=None, stay_away=None):
        return FakeContent(self.share_value)


class FakeContent:
    """The content check, from the fake's good-place share: little open ground, mostly things not to look at."""

    def __init__(self, share):
        self.share = share

    def unwanted(self, vector):
        return 1.0 - self.share

    def dull(self, vector):
        return max(0.0, 0.8 - self.share)


def _looking_hunter(embedder, asked):
    ptz = FakePtz(View(131.0, -10.0, 36.9))

    def ask(camera_id, jpeg, questions):
        asked.append(sorted(questions) if not isinstance(jpeg, list) else ["tiles"])
        return {q: {"true": 0.9 if q == "_ground" else 0.02} for q in questions if q != "_kind"} | {"_kind": kind(ground=1.0)}

    hunter = Hunter(HunterConfig("cam", cues=["Is there a path or walkway?"]), ptz, ask, crop=no_crop,
                    sleep=ptz.sleep, clock=lambda: ptz.now, embedder=embedder)
    hunter._expected = ptz.position()
    return hunter, ptz.position()


def test_a_view_the_embedder_rates_hopeless_skips_clefs_scene_questions_and_tiles():
    asked = []
    hunter, view = _looking_hunter(FakeEmbedder(share=0.04), asked)  # the side of the house
    hunter._look(survey=True, view=view)
    assert asked == [["t0"]]  # only the hunt question: no scene questions, no tiles
    stats = hunter.stats[view]
    assert hopeless(stats) and stats.scored and stats.embed == 0.04
    other = View(190.0, -10.0, 36.9)
    hunter.stats[other] = ViewStats(prior=0.3)
    assert choose_next(hunter.stats, view, now=1000.0, hour=10) == other


def test_a_promising_view_is_scored_by_embeddings_and_clef_confirms():
    asked = []
    hunter, view = _looking_hunter(FakeEmbedder(share=0.6), asked)
    hunter._look(survey=True, view=view)
    assert asked[0] == ["t0"] and ["tiles"] in asked  # no scene questions; Clef asks the hunt question and the tiles
    stats = hunter.stats[view]
    assert stats.embed == 0.6 and stats.scored and not hopeless(stats) and stats.prior > 0.2
    assert stats.unwanted == pytest.approx(0.4)  # 40% things not to look at: promising, but less so
    assert set(stats.likeness) == {"Is there a person in view?", "Is there a path or walkway?"}


def test_with_the_embedder_down_clef_scores_the_view_as_before():
    asked = []
    hunter, view = _looking_hunter(FakeEmbedder(down=True), asked)
    hunter._look(survey=True, view=view)
    assert any(q.startswith("_") for q in asked[0]) and hunter.stats[view].embed == -1.0


def test_the_tile_scan_uses_embeddings_and_skips_hopeless_tiles():
    sims = [{"Is there a path or walkway?": 0.62} for _ in range(9)]
    sims[4] = {"Is there a path or walkway?": 0.72}  # a path stands out in the middle tile
    sims[0] = {"Is there a path or walkway?": 0.71}  # and in a corner that is mostly the house wall
    shares = [0.5] * 9
    shares[0] = 0.03
    asked = []
    hunter, view = _looking_hunter(FakeEmbedder(tile_shares=shares, tile_sims=sims), asked)
    hunter.home = view
    hunter.config.camera_height_m = 40.0
    hunter.stats[view] = ViewStats(prior=0.6, scored=True, scene=0.6, cue_scan_at=-1e9)  # due for a scan
    hunter._scan_tiles(view, b"jpeg")
    children = [(v, s) for v, s in hunter.stats.items() if v.fov < view.fov]
    assert asked == [] and len(children) == 1  # no Clef request; the wall corner did not become a view
    child, stats = children[0]
    assert stats.found_by == "Is there a path or walkway?" and stats.embed == 0.5


def test_good_share_and_standout_tiles():
    from clef_operator.embed import good_share, standout_tiles
    good, bad = [[1.0, 0.0]], [[0.0, 1.0]]
    assert good_share([0.9, 0.1], good, bad) > 0.99 and good_share([0.1, 0.9], good, bad) < 0.01
    tiles = [{"path": 0.62}] * 4 + [{"path": 0.73}] + [{"path": 0.63}] * 4
    ((index, score, cue),) = standout_tiles(tiles)
    assert index == 4 and cue == "path" and score == pytest.approx(1.0)


def test_descriptions_become_clef_questions_and_questions_stay():
    from clef_operator.hunter import as_question
    assert as_question("a person") == "Is there a person in view?"
    assert as_question(" a delivery van ") == "Is there a delivery van in view?"
    assert as_question("Is there a person in view?") == "Is there a person in view?"


def test_views_are_ranked_by_how_much_more_they_look_like_the_target_than_the_usual_view():
    from clef_operator.hunter import embedded_prior
    cfg = HunterConfig("cam", targets=["a person"], cues=["a picnic table"], avoid=["trees"])
    usual = {"a person": 0.62, "a picnic table": 0.62}
    views = [ViewStats(embed=0.6, likeness=dict(usual), unlike={"trees": 0.70}) for _ in range(5)]
    picnic = ViewStats(embed=0.6, likeness={"a person": 0.64, "a picnic table": 0.70}, unlike={"trees": 0.70})
    forest = ViewStats(embed=0.5, likeness=dict(usual), unlike={"trees": 0.80})
    like_m = camera_medians([v.likeness for v in views + [picnic, forest]])
    unlike_m = camera_medians([v.unlike for v in views + [picnic, forest]])
    avoid_usual, prior_usual = embedded_prior(views[0], cfg, like_m, unlike_m)
    avoid_picnic, prior_picnic = embedded_prior(picnic, cfg, like_m, unlike_m)
    avoid_forest, prior_forest = embedded_prior(forest, cfg, like_m, unlike_m)
    assert prior_picnic > prior_usual > prior_forest
    assert avoid_usual < 0.1  # trees in every view of the yard: not a reason to avoid them all
    assert avoid_forest >= 0.75  # mostly trees, far more than the usual view



def test_close_ups_are_for_the_distance_zoomed_to_suit_it():
    from clef_operator.hunter import distant_zoom, target_degrees
    cfg = HunterConfig("cam")  # 3.5 m up, looking for a person
    assert target_degrees(-15.0, 3.5, 1.7) > 7.0  # 13 m away: already big in a 37 degree view
    assert distant_zoom(View(190.0, -15.0, 37.0), cfg, 37.0) is None  # so no close-up of the grass there
    mid, far = distant_zoom(View(190.0, -5.0, 37.0), cfg, 37.0), distant_zoom(View(190.0, -1.0, 37.0), cfg, 37.0)
    assert 15.0 < mid < 25.0 and far == cfg.min_fov  # farther: zoomed in further


def test_the_look_around_skips_close_ups_of_the_near_ground():
    now = [1000.0]
    hunter = Hunter(HunterConfig("cam"), FakePtz(), lambda *a: {}, clock=lambda: now[0])
    hunter.home = View(184.0, -12.0, 37.0)
    wide = View(190.0, -10.0, 37.0)
    near_grass, far_gate = View(195.0, -15.0, 9.0), View(175.0, -3.0, 14.0)
    hunter.stats = {wide: ViewStats(prior=0.6), near_grass: ViewStats(prior=0.9), far_gate: ViewStats(prior=0.5)}
    hunter._wide_since = 0.0  # wide long enough to look around
    assert set(hunter._candidates()) == {far_gate}


def test_magnets_are_visited_even_when_they_look_like_something_to_avoid():
    home, yard = View(248.0, -41.5, 37.0), View(190.0, -10.0, 37.0)
    stats = {home: ViewStats(prior=0.3, visits=20, avoid=0.9, scored=True, scene=0.3),  # trees in front of the preset
             yard: ViewStats(prior=0.6, visits=20, last_visit=990.0, scored=True, scene=0.6)}
    pick = choose_next(stats, yard, now=1000.0, hour=10, magnets=[home], since_magnet_s=900.0)
    assert pick == home  # away from it for 15 minutes: back to the magnet, trees or not


def _still_hunter():
    ptz = FakePtz(View(190.0, -5.0, 36.9))
    hunter = Hunter(HunterConfig("cam", settle_s=0.0), ptz, lambda *a: {}, sleep=ptz.sleep, clock=lambda: ptz.now)
    hunter.home = View(190.0, -5.0, 36.9)
    view = View(190.0, -5.0, 37.0)
    hunter.stats[view] = ViewStats(prior=0.5)
    hunter.status.view, hunter.status.state, hunter._arrived_at = view, "holding", -100.0
    return hunter, ptz, view


def test_action_in_the_distance_while_still_is_chased_quickly_and_zoomed_in_on():
    from clef_operator.action import MIN_SEEN
    hunter, ptz, view = _still_hunter()
    t = 0.0
    for _ in range(MIN_SEEN + 10):  # a quiet view, watched a while
        hunter.motion_frame(t, set())
        t += 0.5
    person = {(30, 12), (30, 13), (31, 12), (31, 13)}  # right of centre, just below the horizon: far away
    hunter.motion_frame(t, person)
    hunter.motion_frame(t + 0.5, {(c + 1, r) for c, r in person})
    assert hunter._action is not None and hunter.activity
    ptz.now = t + 1.0
    target = hunter._take_action()
    assert target is not None and target.pan > view.pan and target.fov < 25.0  # to the right, zoomed in
    assert "motion" in hunter.zoom_reason


def test_motion_above_the_horizon_is_treetops_or_sky_and_never_chased():
    from clef_operator.action import MIN_SEEN
    hunter, ptz, view = _still_hunter()  # tilt -5, 37 degrees wide: the top of the frame is about 9 degrees up
    t = 0.0
    for _ in range(MIN_SEEN + 10):
        hunter.motion_frame(t, set())
        t += 0.5
    treetop = {(30, 1), (30, 2), (31, 1), (31, 2)}  # a tree line waving, high in the frame
    hunter.motion_frame(t, treetop)
    hunter.motion_frame(t + 0.5, {(c + 1, r) for c, r in treetop})
    assert hunter._action is None and not hunter.activity
    t += 60.0
    for _ in range(12):
        hunter.motion_frame(t, set())
        t += 0.5
    foot = {(30, 10), (30, 11), (31, 10), (31, 11)}  # the foot of the tree line, centred about on the horizon
    hunter.motion_frame(t, foot)
    hunter.motion_frame(t + 0.5, {(c + 1, r) for c, r in foot})
    assert hunter._action is None and not hunter.activity


def test_motion_that_keeps_coming_back_to_one_spot_with_nobody_there_stops_drawing_chases():
    from clef_operator.action import MIN_SEEN
    hunter, ptz, view = _still_hunter()
    t = 0.0
    for _ in range(MIN_SEEN + 10):
        hunter.motion_frame(t, set())
        t += 0.5
    bush = {(30, 20), (30, 21), (31, 20), (31, 21)}  # below the horizon: a bush in the wind
    chases = 0
    for gust in range(4):
        t += 120.0  # a gust every two minutes, quiet in between
        for _ in range(12):
            hunter.motion_frame(t, set())
            t += 0.5
        hunter._action = None
        hunter.motion_frame(t, bush)
        hunter.motion_frame(t + 0.5, {(c + 1, r) for c, r in bush})
        chases += hunter._action is not None
        t += 1.0
    assert chases == 2  # twice, then it is known as the wind
    # Someone found there since: it counts again.
    hunter.stats[view].last_yes = t
    t += 120.0
    for _ in range(12):
        hunter.motion_frame(t, set())
        t += 0.5
    hunter.motion_frame(t, bush)
    hunter.motion_frame(t + 0.5, {(c + 1, r) for c, r in bush})
    assert hunter._action is not None


def test_fruitless_chases_are_remembered_across_a_restart(tmp_path):
    ptz = FakePtz(View(190.0, -5.0, 36.9))
    memory = tmp_path / "hunter.json"
    hunter = Hunter(HunterConfig("cam", settle_s=0.0), ptz, lambda *a: {}, sleep=ptz.sleep, clock=lambda: ptz.now,
                    memory=memory)
    hunter.stats[View(190.0, -5.0, 37.0)] = ViewStats(prior=0.5)
    hunter.nuisance.chased(100.0, 200.0, -22.0)
    hunter.nuisance.chased(200.0, 200.5, -22.5)
    hunter._save()
    again = Hunter(HunterConfig("cam", settle_s=0.0), ptz, lambda *a: {}, sleep=ptz.sleep, clock=lambda: ptz.now,
                   memory=memory)
    again._load()
    assert again._fruitless(300.0, View(200.0, -22.0, 37.0))  # the leaves there are still known after the restart
    assert not again._fruitless(300.0, View(150.0, -10.0, 37.0))
    # Below the camera a bush spans many degrees: chases 7 degrees apart there are the same place; out in the yard not.
    again.nuisance.chased(400.0, 238.0, -26.5)
    again.nuisance.chased(500.0, 231.5, -33.0)
    assert again._fruitless(600.0, View(237.0, -31.0, 37.0))
    again.nuisance.chased(400.0, 150.0, -4.0)
    again.nuisance.chased(500.0, 157.0, -4.0)
    assert again._fruitless(600.0, View(153.5, -4.0, 37.0))  # within 5 degrees of both
    assert not again._fruitless(600.0, View(145.5, -4.0, 37.0))  # out in the yard the radius stays 5 degrees: one chase


def test_motion_while_the_camera_moves_or_is_handed_over_is_ignored():
    hunter, ptz, view = _still_hunter()
    hunter._handed_over_until = 1e9
    for i in range(200):
        hunter.motion_frame(i * 0.5, {(30, 12), (31, 12)} if i > 100 else set())
    assert hunter._watching is None and hunter._action is None


def test_a_camera_detection_is_a_find_and_hands_the_camera_to_its_auto_tracking():
    hunter, ptz, view = _still_hunter()
    hunter.camera_detected(1000.0, "Person", "track-1", last_seen=1000.0)
    assert hunter.stats[view].total_hits == 1 and hunter._yield
    assert hunter._handed_over_until == 1000.0 + max(hunter.config.handback_s, TRACK_HOLD_S)
    hunter.camera_detected(1030.0, "Person", "track-1", last_seen=1030.0)  # still tracking it
    assert hunter._handed_over_until == 1030.0 + max(hunter.config.handback_s, TRACK_HOLD_S) and hunter.stats[view].total_hits == 1


def test_a_move_under_way_stops_when_the_camera_detects_the_target():
    ptz = FakePtz(View(190.0, -5.0, 36.9))
    calls = [0]

    def detected():
        calls[0] += 1
        return calls[0] > 4  # the camera sees a person a second into the move
    with pytest.raises(ExternalControl):
        move_slowly(ptz, View(250.0, -5.0, 36.9), max_speed=0.2, sleep=ptz.sleep, abort=detected)
    assert ptz.speed == (0.0, 0.0, 0.0)  # its own move stopped, so the tracker can take the camera


def test_recent_action_pulls_the_choice_of_view_more_at_night():
    lawn, fence = View(190.0, -10.0, 37.0), View(250.0, -10.0, 37.0)
    stats = {lawn: ViewStats(prior=0.5, visits=20, last_visit=900.0), fence: ViewStats(prior=0.5, visits=20, last_visit=900.0)}
    activity = [(990.0, View(252.0, -8.0, 37.0))]  # something moved by the fence ten seconds ago
    kw = dict(now=1000.0, hour=22, exploration=0.0, activity=activity)
    assert choose_next(stats, View(220.0, -10.0, 37.0), activity_weight=0.8, **kw) == fence


def test_the_cameras_detections_are_matched_to_what_the_hunter_looks_for():
    from clef_operator.hunter import detection_kind
    assert detection_kind("nx.milesight.Person", ["person"]) == "Person"
    assert detection_kind("nx.milesight.Person", ["Is there a person in view?"]) == "Person"
    assert detection_kind("nx.milesight.Person", ["someone at the gate"]) == "Person"
    assert detection_kind("nx.milesight.Vehicle", ["a delivery van"]) == "Vehicle"
    assert detection_kind("nx.milesight.Vehicle", ["person"]) is None
    assert detection_kind("nx.milesight.Person", ["a parked car"]) is None



def test_a_wide_view_is_watched_a_while_for_motion_and_action_cuts_it_short():
    ptz = FakePtz()
    hunter = Hunter(HunterConfig("cam"), ptz, lambda *a: {}, sleep=ptz.sleep, clock=lambda: ptz.now)
    hunter._arrived_at = ptz.now
    hunter._hold()
    assert ptz.now >= 20.0  # no windows coming, but wide: it watched (20 s by day, 45 at night)
    hunter._arrived_at = ptz.now
    hunter._action = (ptz.now, View(190.0, -10.0, 37.0), (0.4, 0.4, 0.1, 0.1))
    start = ptz.now
    hunter._hold()
    assert ptz.now == start  # action: off to catch it at once



def test_the_content_check_sorts_what_not_to_look_at_from_open_ground():
    from clef_operator.embed import ContentModel
    def unit(*xs):
        n = sum(x * x for x in xs) ** 0.5
        return [x / n for x in xs] + [0.0] * 253  # padded to the 256 dims the check uses
    prompts = {"tree": [unit(1, 0, 0)], "grass": [unit(0, 1, 0)], "yard": [unit(0, 0, 1)]}
    examples = [(unit(0.9, 0.1, 0), "tree"), (unit(0.1, 0.9, 0.1), "grass"), (unit(0, 0.2, 0.9), "yard")]
    model = ContentModel(prompts, examples)
    assert model.unwanted(unit(0.95, 0.05, 0)) > 0.9  # leaves
    assert model.unwanted(unit(0, 0.3, 0.95)) < 0.1 and model.dull(unit(0, 0.3, 0.95)) < 0.1  # a fence line
    assert model.unwanted(unit(0.05, 0.95, 0.1)) < 0.1 and model.dull(unit(0.05, 0.95, 0.1)) > 0.9  # plain lawn


def test_no_close_ups_of_grass_or_trees_on_the_look_around():
    now = [1000.0]
    hunter = Hunter(HunterConfig("cam", camera_height_m=40.0), FakePtz(), lambda *a: {}, clock=lambda: now[0])
    hunter.home = View(184.0, -12.0, 37.0)
    wide = View(190.0, -10.0, 37.0)
    lawn, gate = View(195.0, -5.0, 9.0), View(175.0, -3.0, 14.0)
    hunter.stats = {wide: ViewStats(prior=0.6), lawn: ViewStats(prior=0.9, dull=0.9), gate: ViewStats(prior=0.5, dull=0.1)}
    hunter._wide_since = 0.0
    assert set(hunter._candidates()) == {gate}


def test_close_ups_learned_before_the_content_check_are_classified_from_the_map():
    from clef_operator.embed_map import EmbedMap
    emap = EmbedMap()
    lawn = View(195.0, -5.0, 9.0)
    emap.add_look(lawn, [1.0, 0.0], [], [], b"", now=1.0)
    hunter = Hunter(HunterConfig("cam"), FakePtz(), lambda *a: {}, embedder=FakeEmbedder(share=0.1), embed_map=emap)
    hunter.home = View(184.0, -12.0, 37.0)
    hunter.stats[lawn] = ViewStats(prior=0.9)
    hunter._classify_close_ups()
    assert hunter.stats[lawn].dull == pytest.approx(0.7)  # the fake: mostly lawn



def test_two_neighbouring_lawn_views_do_not_pull_the_hunter_back_and_forth():
    # 09:53 on Back PTZ: home (184/-12, a magnet) and 175/-10 (three finds) alternated every 15 s, 8 degrees apart.
    home, neighbour, fence = View(184.0, -12.0, 37.0), View(175.0, -10.0, 37.0), View(249.0, -30.0, 37.0)
    stats = {home: ViewStats(prior=0.5, visits=50, last_visit=990.0),
             neighbour: ViewStats(prior=0.5, visits=50, last_visit=970.0, hits={9: 3}),
             fence: ViewStats(prior=0.45, visits=50, last_visit=700.0)}
    kw = dict(now=1000.0, hour=9, exploration=0.0, magnets=[home], since_magnet_s=10.0)
    assert choose_next(stats, home, **kw) == fence


def test_every_worthwhile_wide_view_is_watched_but_not_hopeless_ones():
    hunter = Hunter(HunterConfig("cam"), FakePtz(), lambda *a: {})
    assert hunter._worth_watching(ViewStats(prior=0.1))
    assert not hunter._worth_watching(ViewStats(avoid=0.9))
    assert not hunter._worth_watching(ViewStats(scored=True, unwanted=0.9))


def test_a_close_up_read_back_tighter_than_asked_is_learned_at_the_zoom_asked_for():
    class TightReadback(FakePtz):  # Back PTZ: asked for 6 degrees, it reads about 3.5
        def position(self):
            v = super().position()
            return View(v.pan, v.tilt, 3.5) if v.fov < 7.0 else v
    ptz = TightReadback(View(175.0, -4.0, 36.9))
    hunter = Hunter(HunterConfig("cam", settle_s=0.0, focus_s=0.0), ptz, lambda *a: {}, sleep=ptz.sleep,
                    clock=lambda: ptz.now)
    hunter.home = View(184.0, -12.0, 36.9)
    arrived = hunter._go(View(175.0, -4.0, 6.0))
    assert arrived.fov == 6.0 and not any(v.fov < 6.0 for v in hunter.stats)


def test_views_left_tighter_than_the_hunter_zooms_are_dropped_at_start(tmp_path):
    memory = tmp_path / "hunter.json"
    rows = [{"pan": 175.0, "tilt": -4.0, "fov": 3.5, "stats": ViewStats(visits=20).to_json()},
            {"pan": 176.0, "tilt": -4.0, "fov": 4.0, "stats": ViewStats(visits=2, hits={9: 1}).to_json()},
            {"pan": 184.0, "tilt": -12.0, "fov": 37.0, "stats": ViewStats(visits=50).to_json()}]
    memory.write_text(json.dumps({"version": MEMORY_VERSION, "home": None, "views": rows}))
    hunter = Hunter(HunterConfig("cam"), FakePtz(), lambda *a: {}, memory=memory)
    hunter._load()
    assert {v.fov for v in hunter.stats} == {4.0, 37.0}  # the one with a find is kept


def test_the_cameras_tracking_is_wanted_only_while_it_follows_what_it_detected():
    hunter, ptz, view = _still_hunter()
    assert not hunter.tracking_wanted(1000.0)  # hunting: tracking off, or panning sets it chasing the scene
    hunter.camera_detected(1000.0, "Person", "t1", last_seen=1000.0)
    assert hunter.tracking_wanted(1000.0 + hunter.config.handback_s + 5.0)  # tracks last seconds: hold on past them
    assert not hunter.tracking_wanted(1000.0 + TRACK_HOLD_S + 1.0)  # lost them: back to hunting


def test_the_cameras_tracking_stays_on_while_the_camera_follows_someone_after_their_track_ends():
    ptz = FakePtz(View(190.0, -5.0, 36.9))
    hunter = Hunter(HunterConfig("cam", settle_s=0.0, handback_s=10.0), ptz, lambda *a: {}, sleep=ptz.sleep,
                    clock=lambda: ptz.now)
    hunter._expected = ptz.position()
    hunter.camera_detected(ptz.now, "Person", "t1", last_seen=ptz.now)  # a 2 s track, then nothing more from Nx
    start = ptz.now
    ptz.speed = (0.03, 0.0, 0.0)  # the camera's tracker following them across the yard, for 90 s
    for _ in range(45):
        ptz.sleep(2.0)
        hunter._watch(ptz.now)
        assert hunter.tracking_wanted(ptz.now) and ptz.now < hunter._handed_over_until  # never cut off mid-follow
    assert ptz.now - start >= 90.0 and ptz.speed != (0.0, 0.0, 0.0)  # and not stopped as a runaway after 60 s
    ptz.speed = (0.0, 0.0, 0.0)  # it lost them and stopped
    for _ in range(20):
        ptz.sleep(2.0)
        hunter._watch(ptz.now)
    assert not hunter.tracking_wanted(ptz.now) and ptz.now >= hunter._handed_over_until  # back to hunting


def test_after_stopping_a_move_nobody_asked_for_the_hunter_stops_holding_and_moves_on():
    ptz = FakePtz(View(200.0, -10.0, 36.9))
    hunter = Hunter(HunterConfig("cam", handback_s=10.0), ptz, lambda *a: {}, sleep=ptz.sleep, clock=lambda: ptz.now)
    hunter.status.view = View(184.0, -12.0, 37.0)
    hunter._handed_over(ptz.now, ptz.position())
    ptz.speed = (0.05, 0.0, 0.0)  # the camera's tracker panning off somewhere
    for _ in range(32):
        ptz.sleep(2.0)
        hunter._watch(ptz.now)
    assert ptz.speed == (0.0, 0.0, 0.0) and hunter.status.view is None and hunter._lost
    hunter._arrived_at = ptz.now
    start = ptz.now
    hunter._hold()
    assert ptz.now == start  # no holding on at a view it is not at



def test_old_close_ups_much_tighter_than_their_distance_needs_are_skipped():
    from clef_operator.hunter import distant_zoom
    now = [1000.0]
    hunter = Hunter(HunterConfig("cam"), FakePtz(), lambda *a: {}, clock=lambda: now[0])
    hunter.home = View(184.0, -12.0, 37.0)
    wide = View(190.0, -10.0, 37.0)
    right = max(14.0, distant_zoom(View(169.0, -3.4, 14.0), hunter.config, 37.0))  # about 14 degrees, 57 m away
    too_tight, fitting = View(169.0, -3.4, 6.0), View(175.0, -3.4, round(right, 1))
    hunter.stats = {wide: ViewStats(prior=0.6), too_tight: ViewStats(prior=0.9), fitting: ViewStats(prior=0.5)}
    hunter._wide_since = 0.0
    assert set(hunter._candidates()) == {fitting}


def test_a_zoom_is_one_timed_command_even_with_a_late_readback_and_never_pumps():
    ptz = FakePtz(View(172.0, -3.0, 36.9), lag=1.5)  # Back PTZ reads back a second or more late
    zooms = []
    real = ptz.continuous
    def continuous(pan, tilt, zoom):
        if zoom:
            zooms.append(zoom)
        real(pan, tilt, zoom)
    ptz.continuous = continuous
    hunter = Hunter(HunterConfig("cam", settle_s=0.0, focus_s=0.0), ptz, lambda *a: {}, sleep=ptz.sleep,
                    clock=lambda: ptz.now)
    hunter._go(View(172.0, -3.0, 14.0))
    assert zooms == [zooms[0]] and zooms[0] > 0  # one zoom in, never back out
    assert abs(ptz.fov - 14.0) / 14.0 < 0.15
    assert abs(zoom_seconds(36.9, 12.7) - 2.5) < 0.1  # measured on Back PTZ: 36.9 to 12.7 degrees in 2.5 s
    zooms.clear()
    hunter._go(View(172.0, -3.0, 13.0))  # within 15%: close enough, no zoom at all
    assert zooms == []
