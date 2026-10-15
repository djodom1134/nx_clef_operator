import base64
import threading
import time

import pytest

from clef_operator import gateway as gw
from clef_operator.clef import ClefResult


class FakeClef:
    """Answers instantly except the first request, which blocks until released, so jobs pile up behind it."""

    url = "fake"

    def __init__(self) -> None:
        self.release = threading.Event()
        self.order: list[str] = []
        self.states: list[str] = []
        self.requests: list[dict] = []
        self.fps: list = []

    def health(self) -> dict:
        return {"status": "ready"}

    def ask(self, state, questions, videos=None, video_fps=None):
        self.order.append(next(iter(questions)))
        self.states.append(state)
        self.requests.append(dict(questions))
        self.fps.append(video_fps)
        if len(self.order) == 1:
            self.release.wait(5)
        return ClefResult({q: {"true": 0.9, "false": 0.1} for q in questions}, 1000, 0.01)


@pytest.fixture
def gateway(monkeypatch):
    monkeypatch.setattr(gw, "prepare_clip", lambda clip, *args, **kwargs: clip)
    monkeypatch.setattr(gw, "GATHER_S", 0.0)  # one job per window in these tests: nothing to gather
    fake = FakeClef()
    return gw.Gateway(fake, gw.MAX_PIXELS), fake


@pytest.fixture
def bundling(monkeypatch):
    """A gateway that gathers each window's jobs, recording the frame rate and zone each clip was prepared at."""
    clips = []
    monkeypatch.setattr(gw, "prepare_clip", lambda clip, fps, width, box, *a, **k: clips.append((fps, box)) or clip)
    monkeypatch.setattr(gw, "GATHER_S", 0.2)
    fake = FakeClef()
    fake.release.set()
    return gw.Gateway(fake, gw.MAX_PIXELS), fake, clips


def job(qid, priority, camera="cam", end=None, **extra):
    return {"camera_id": camera, "camera_name": "Front porch", "priority": priority, "fps": 2,
            "window_end": end or time.time(), "questions": {qid: {"type": "noul", "instructions": qid}},
            "clip": base64.b64encode(b"clip").decode(), **extra}


def submit_async(g, body, results, key):
    t = threading.Thread(target=lambda: results.__setitem__(key, g.submit(body)))
    t.start()
    return t


def test_critical_jumps_the_queue(gateway):
    g, fake = gateway
    results = {}
    threads = [submit_async(g, job("blocker", "normal", camera="a"), results, "blocker")]
    time.sleep(0.2)  # the blocker is now running
    threads.append(submit_async(g, job("routine", "normal", camera="b"), results, "routine"))
    time.sleep(0.1)
    threads.append(submit_async(g, job("fight", "critical", camera="c"), results, "fight"))
    time.sleep(0.1)
    fake.release.set()
    for t in threads:
        t.join(5)
    assert fake.order == ["blocker", "fight", "routine"]
    assert all(r["status"] == "answered" for r in results.values())


def test_merged_job_caller_is_told(gateway):
    g, fake = gateway
    results = {}
    threads = [submit_async(g, job("blocker", "critical", camera="x"), results, "blocker")]
    time.sleep(0.2)
    now = time.time()
    threads.append(submit_async(g, job("old", "low", camera="y", end=now - 5), results, "old"))
    time.sleep(0.1)
    threads.append(submit_async(g, job("new", "low", camera="y", end=now), results, "new"))
    time.sleep(0.8)  # the sweeper releases the merged job's caller
    fake.release.set()
    for t in threads:
        t.join(5)
    assert results["old"] == {"status": "dropped", "reason": "merged"}
    assert results["new"]["status"] == "answered"
    assert "old" not in fake.order


def test_state_carries_camera_zone_site_and_time():
    state = gw.build_state({
        "camera_id": "x", "camera_name": "Front porch", "camera_description": "Covered porch facing the street.",
        "zone": {"name": "Door", "description": "front door"},
        "site": {"interests": "people after 22:00", "nuisance": ["flag on the pole", "trees"]},
        "window_end": 1792014600,
    }, now=0)
    assert "Front porch" in state and "Covered porch" in state and "zone 'Door'" in state
    assert "people after 22:00" in state and "flag on the pole; trees" in state
    assert "local time" in state


def test_hourly_survey_rides_along_and_is_stripped(gateway):
    g, fake = gateway
    fake.release.set()
    first = g.submit(job("q", "normal", camera="porch"))
    second = g.submit(job("q", "normal", camera="porch"))
    assert set(first["answers"]) == {"q"} and set(second["answers"]) == {"q"}
    assert g.profiles._slots["porch"]  # the first request carried the survey and was tallied
    assert sum(1 for state in fake.states) == 2


def test_codec_names_from_the_server_map_to_ffmpeg_formats():
    assert gw.ffmpeg_format("video/hevc") == "hevc"
    assert gw.ffmpeg_format("video/h264") == "h264"
    assert gw.ffmpeg_format("H265") == "hevc"
    with pytest.raises(ValueError):
        gw.ffmpeg_format("video/mjpeg")


def test_a_normal_job_never_merges_away_a_critical_one_at_the_same_fps(gateway):
    g, fake = gateway
    results = {}
    threads = [submit_async(g, job("blocker", "critical", camera="z"), results, "blocker")]
    time.sleep(0.2)
    now = time.time()
    threads.append(submit_async(g, job("fight", "critical", camera="p", end=now - 5), results, "fight"))
    time.sleep(0.1)
    threads.append(submit_async(g, job("car", "normal", camera="p", end=now), results, "car"))
    time.sleep(0.8)
    fake.release.set()
    for t in threads:
        t.join(5)
    assert results["fight"]["status"] == "answered" and results["car"]["status"] == "answered"


def test_one_window_goes_to_clef_once_per_zone(bundling):
    # Front panorama: a Critical and a Normal question on the whole frame, and a 1 fps driveway question
    g, fake, clips = bundling
    end = time.time()
    driveway = {"name": "Driveway", "box": [0.5, 0.5, 0.5, 0.5]}
    results, threads = {}, []
    for key, body in {"fight": job("fight", "critical", camera="pano", end=end, fps=4),
                      "car": job("car", "normal", camera="pano", end=end, fps=2),
                      "parked": job("parked", "normal", camera="pano", end=end, fps=1, zone=driveway)}.items():
        threads.append(submit_async(g, body, results, key))
    for t in threads:
        t.join(5)
    assert len(fake.requests) == 2  # whole frame once (not once per priority), the driveway crop once
    whole = next(r for r in fake.requests if "fight" in r)
    assert {"fight", "car"} <= set(whole) and "parked" not in whole
    assert sorted(clips, key=lambda c: c[0]) == [(1.0, [0.5, 0.5, 0.5, 0.5]), (4.0, None)]  # highest fps of the zone
    assert set(results["fight"]["answers"]) == {"fight"} and set(results["car"]["answers"]) == {"car"}
    assert results["car"]["priority"] == "critical" and results["car"]["bundled"] == 2  # rides with the Critical one


def test_bundled_callers_keep_their_own_answers_when_ids_clash(bundling):
    g, fake, _ = bundling
    end = time.time()
    results, threads = {}, []
    for key, priority in (("a", "critical"), ("b", "high")):
        body = job("q1", priority, camera="garage", end=end)
        body["questions"]["q1"]["instructions"] = key
        threads.append(submit_async(g, body, results, key))
    for t in threads:
        t.join(5)
    assert len(fake.requests) == 1 and {"q1", "q1~1"} <= set(fake.requests[0])  # renamed for Clef only
    assert set(results["a"]["answers"]) == {"q1"} and set(results["b"]["answers"]) == {"q1"}


def test_other_cameras_and_windows_are_not_bundled(bundling):
    g, fake, _ = bundling
    end = time.time()
    results, threads = {}, []
    for key, camera, window_end in (("a", "pano", end), ("b", "garage", end), ("c", "pano", end - 20)):
        # Critical: never merged, so the older pano window cannot be replaced by the newer one while queued
        threads.append(submit_async(g, job(key, "critical", camera=camera, end=window_end), results, key))
    for t in threads:
        t.join(5)
    assert len(fake.requests) == 3 and all(r["status"] == "answered" for r in results.values())


def test_a_bad_clip_fails_every_bundled_caller(bundling, monkeypatch):
    g, _, _ = bundling

    def broken(*args, **kwargs):
        raise ValueError("ffmpeg exit 1")

    monkeypatch.setattr(gw, "prepare_clip", broken)
    end = time.time()
    errors = {}

    def call(key, priority):
        try:
            g.submit(job(key, priority, camera="pano", end=end))
        except ValueError as error:
            errors[key] = error

    threads = [threading.Thread(target=call, args=(k, p)) for k, p in (("a", "critical"), ("b", "normal"))]
    for t in threads:
        t.start()
    for t in threads:
        t.join(5)
    assert set(errors) == {"a", "b"}  # the plugin gets the same HTTP 400 it would have alone


def test_a_job_that_cannot_make_its_target_is_sent_at_a_lower_frame_rate(bundling):
    g, fake, clips = bundling
    g.scheduler.cost.per_token_s = 0.002  # a slow Clef: a 30 s window at 4 fps would take about 19 s, target 16 s
    result = g.submit(job("car", "normal", camera="pano", fps=4, window_start=time.time() - 30))
    assert result["status"] == "answered" and result["fps"] < 4 and fake.fps == [result["fps"]]
    assert clips[-1][0] == result["fps"]  # re-encoded at the lower rate before it went to Clef
    assert g.status()["lowered_fps"] == {"NORMAL": 1}


def test_status_reports_latency_per_class_against_its_target(gateway):
    g, fake = gateway
    fake.release.set()
    g.submit(job("q", "low", camera="porch"))
    low = g.status()["latency_1h"]["low"]
    assert low["n"] == 1 and low["on_time"] == 1.0 and low["target_s"] == 20.0 and low["max"] < 5
    porch = g.status()["latency_by_camera_1h"]["Front porch"]
    assert porch["n"] == 1 and porch["mean"] < 5 and porch["on_time"] == 1.0


def test_a_running_hunter_survives_settings_resent_while_the_server_cannot_drive_its_ptz(gateway, monkeypatch):
    g, _ = gateway

    class Running:
        status = type("S", (), {"state": "hunting"})()
        config = gw.HunterConfig("ptz", rest_s=5)

        def stop(self, **kwargs):
            raise AssertionError("must not be stopped")

    g.hunters["ptz"] = Running()
    g._nx = type("Nx", (), {"ptz_operational": lambda self, camera_id: False})()
    result = g.set_hunter({"camera_id": "ptz", "enabled": True, "targets": ["Is there a person in view?"]})
    assert result["status"] == "hunting" and "warning" in result and "ptz" in g.hunters
