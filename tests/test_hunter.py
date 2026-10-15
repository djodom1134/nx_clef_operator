from clef_operator.hunter import (
    CUE_RESCAN_S, DEFAULT_CUES, FOV_RATE, MEMORY_VERSION, PAN_RATE, cue_boost, standout_tiles, zoom_on, QUIET_RECHECK_S, is_quiet, resting, sightings, TILT_RATE, DEFAULT_AVOID, Hunter, HunterConfig, View, ViewStats, aim_at,
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
        self.max_speed_used = max(self.max_speed_used, abs(pan), abs(tilt), abs(zoom))

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
        self.fov -= self.speed[2] * FOV_RATE * seconds  # zoom in narrows the field of view
        self.now += seconds


def no_crop(jpeg, x, y, w, h):
    return f"tile {x:.2f},{y:.2f}".encode()


def test_move_slowly_caps_speed_wraps_pan_and_stops_on_target():
    ptz = FakePtz(View(350.0, -12.0, 36.9))
    end = move_slowly(ptz, View(10.0, -5.0, 20.0), max_speed=0.1, sleep=ptz.sleep)
    assert abs(angle_delta(end.pan, 10.0)) < 1.0 and abs(end.tilt + 5.0) < 1.0 and abs(end.fov - 20.0) < 1.5
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
    assert abs(angle_delta(end.pan, 72.0)) < 1.0 and abs(end.tilt - 3.0) < 1.0 and ptz.speed == (0.0, 0.0, 0.0)
    assert all(v.distance(View(72.0, 3.0, 36.9)) < 1e-6 for v in ptz.absolute_moves)  # only a last small hop
    assert ptz.now < 60


def test_move_stops_at_the_tilt_end_stop_and_undoes_the_pan_it_caused():
    ptz = FakePtz(View(160.0, -25.0, 36.9), tilt_max=5.0)
    end = move_slowly(ptz, View(160.0, 10.0, 36.9), max_speed=0.2, sleep=ptz.sleep)
    assert end.tilt == 5.0 and ptz.speed == (0.0, 0.0, 0.0)
    assert abs(angle_delta(end.pan, 160.0)) < 1.0  # pan drifted while tilt was stuck, then was put back
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
    assert choose_next(stats, here, now=3000.0, hour=9, magnets=[neighbour], since_magnet_s=40.0,
                       travel_weight=0.01) == neighbour  # the old travel cost: 79 degrees was never worth it


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
    assert hunter.status.state == "stopped" and abs(angle_delta(ptz.pan, 184.0)) < 1.0  # back home


def test_an_empty_view_asks_all_tiles_in_one_request():
    ptz = FakePtz()
    asked = []

    def ask(camera_id, jpeg, questions):
        asked.append(len(jpeg) if isinstance(jpeg, list) else 1)
        if len(asked) >= 2:
            hunter.stop(wait=False)
        return {q: {"true": 0.02} for q in questions if q != "_kind"} | {"_kind": kind(ground=1.0)}

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
    assert abs(angle_delta(ptz.pan, 183.9)) < 1.0 and abs(ptz.tilt + 12.0) < 1.0  # stopped: went to the real home
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
    cfg = HunterConfig("cam", cues=["path", "seat", "yard"])
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
    assert 1 <= len(zoomed) <= 3 and all(v.fov == 18.5 for v, _ in zoomed)  # half the field of view
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


def test_zooming_is_one_quick_step_after_a_slow_pan_and_then_the_camera_refocuses():
    ptz = FakePtz(View(184.0, -12.0, 37.0))
    hunter = Hunter(HunterConfig("cam", focus_s=4.0, settle_s=1.0, rest_s=0.0), ptz, lambda *a: {},
                    sleep=ptz.sleep, clock=lambda: ptz.now)
    arrived = hunter._go(View(200.0, -15.0, 18.5))
    assert arrived == View(200.0, -15.0, 18.5)
    absolute_times = [t for t, kind in ptz.commands if kind == "absolute"]
    zoom_index = next(i for i, v in enumerate(ptz.absolute_moves) if v.fov == 18.5)
    zoom, zoom_at = ptz.absolute_moves[zoom_index], absolute_times[zoom_index]
    assert abs(zoom.pan - 200.0) < 1.0 and abs(zoom.tilt + 15.0) < 1.0  # centred first (slowly), then zoomed
    assert all(v.fov == 37.0 for v in ptz.absolute_moves[:zoom_index])  # the pan itself stayed wide
    assert ptz.now - zoom_at >= 4.0 + 1.0  # focus and settle before anything else happens
    before = len(ptz.absolute_moves)
    hunter._go(View(150.0, -10.0, 37.0))
    zoom_out = ptz.absolute_moves[before]
    assert zoom_out.fov == 37.0 and abs(zoom_out.pan - 200.0) < 1.0  # widened first, then the pan


def test_no_commands_are_sent_while_the_camera_refocuses():
    ptz = FakePtz(View(184.0, -12.0, 37.0))
    hunter = Hunter(HunterConfig("cam", focus_s=4.0, settle_s=1.0, rest_s=0.0), ptz, lambda *a: {},
                    sleep=ptz.sleep, clock=lambda: ptz.now)
    hunter._go(View(184.0, -12.0, 18.5))  # zoom only
    zoom_at = ptz.commands[-1][0]
    hunter._go(View(190.0, -12.0, 18.5))
    later = [t for t, _ in ptz.commands if t > zoom_at]
    assert later and min(later) >= zoom_at + 4.0


def test_the_hunter_spends_about_three_quarters_of_its_time_wide_then_gets_bored():
    ptz = FakePtz()
    hunter = Hunter(HunterConfig("cam", wide_share=0.75), ptz, lambda *a: {})
    hunter.home = View(184.0, -12.0, 37.0)
    wide, close = View(190.0, -10.0, 37.0), View(175.0, -5.0, 18.5)
    hunter.stats = {wide: ViewStats(prior=0.6), close: ViewStats(prior=0.6, found_by="play area")}
    hunter._spent(wide, 300.0)
    assert set(hunter._candidates()) == {close}  # all wide lately: look closer
    hunter._spent(close, 200.0)
    assert set(hunter._candidates()) == {wide}  # 60% wide: back out
    hunter.stats = {wide: ViewStats(prior=0.6)}
    hunter.time_wide, hunter.time_zoomed = 300.0, 0.0
    assert set(hunter._candidates()) == {wide}  # nothing zoomed to look at yet: stay wide


def test_zoomed_views_can_be_scanned_for_a_second_zoom_level():
    ptz = FakePtz()
    cfg = HunterConfig("cam", cues=["path", "seat", "yard"])
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
    assert any(v.fov == 9.0 or v.fov == 9.5 for v, s in hunter.stats.items() if s.found_by in ("seat", "path"))
