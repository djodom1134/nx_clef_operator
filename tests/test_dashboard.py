import json
import threading
import time
import urllib.request
from http.server import ThreadingHTTPServer

from clef_operator import gateway as gw
from clef_operator.dashboard import AnswerBoard, answer_row
from test_gateway import FakeClef, job


def test_answers_are_judged_with_the_plugins_thresholds():
    car = answer_row("c1", {"type": "noul", "instructions": "Is a car parked?"}, {"true": 0.85, "false": 0.15},
                     {"on": 0.8}, "", 4.0, "normal", 100.0, 101.0, 3.2)
    assert car["yes"] and car["p"] == 0.85 and car["latency_s"] == 3.2
    quiet = answer_row("c1", {"type": "noul", "instructions": "Is a car parked?"}, {"true": 0.6}, {"on": 0.8},
                       "", 4.0, "normal", 100.0, 101.0, None)
    assert not quiet["yes"]  # below its on-threshold: not an alert yet
    door = answer_row("c4", {"type": "choice", "instructions": "Is the garage door open?",
                             "criteria": {"open": "open", "closed": "closed"}},
                      {"open": 0.15, "closed": 0.85}, {"on": 0.8, "alert_on": ["open"]}, "Door", 2.0, "low", 0, 1, None)
    assert not door["yes"] and door["option"] == "closed" and door["zone"] == "Door"


def test_the_board_keeps_the_latest_answer_per_camera_and_question():
    board = AnswerBoard()
    body = {"camera_id": "{cam}", "camera_name": "Garage", "window_end": 100.0,
            "questions": {"c1": {"type": "noul", "instructions": "Is a person here?"}}, "alerts": {"c1": {"on": 0.8}}}
    board.record(body, {"c1": {"true": 0.1}}, 2.0, "low", 1.5, now=101.0)
    board.record({**body, "window_end": 120.0}, {"c1": {"true": 0.9}}, 2.0, "low", 1.4, now=121.0)
    (camera,) = board.cameras(now=125.0)
    assert camera["name"] == "Garage" and camera["age_s"] == 4.0
    assert len(camera["questions"]) == 1 and camera["questions"][0]["yes"]


def test_the_gateway_serves_the_dashboard_with_each_cameras_answers(monkeypatch):
    monkeypatch.setattr(gw, "prepare_clip", lambda clip, *args, **kwargs: clip)
    monkeypatch.setattr(gw, "GATHER_S", 0.2)
    fake = FakeClef()
    fake.release.set()
    g = gw.Gateway(fake, gw.MAX_PIXELS)
    end = time.time()
    threads = []
    for qid, priority in (("s1", "critical"), ("c1", "normal")):  # one window, bundled into one Clef request
        body = job(qid, priority, camera="pano", end=end, camera_name="Front panorama", alerts={qid: {"on": 0.8}})
        threads.append(threading.Thread(target=g.submit, args=(body,)))
    for t in threads:
        t.start()
    for t in threads:
        t.join(5)
    data = g.dashboard()
    (camera,) = data["cameras"]
    assert camera["name"] == "Front panorama" and {q["id"] for q in camera["questions"]} == {"s1", "c1"}
    assert all(q["yes"] for q in camera["questions"])  # FakeClef answers 0.9

    server = ThreadingHTTPServer(("127.0.0.1", 0), gw.make_handler(g, "secret"))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        page = urllib.request.urlopen(f"{base}/?token=secret", timeout=5).read().decode()
        assert "<title>Clef Operator</title>" in page
        strip = urllib.request.urlopen(f"{base}/strip?camera=pano&token=secret", timeout=5).read().decode()
        assert "<title>Clef status</title>" in strip  # the per-camera strip for Nx layouts
        history = json.load(urllib.request.urlopen(f"{base}/v1/history?camera=pano&hours=1&token=secret", timeout=5))
        assert {q["id"] for q in history["questions"]} == {"s1", "c1"}
        system = json.load(urllib.request.urlopen(f"{base}/v1/system_history?hours=1&token=secret", timeout=5))
        assert system["targets"]["low"] == 20.0 and sum(m["answers"] for m in system["minutes"]) == 1
        payload = json.load(urllib.request.urlopen(f"{base}/v1/dashboard?token=secret", timeout=5))
        assert payload["cameras"][0]["name"] == "Front panorama"
        try:
            urllib.request.urlopen(f"{base}/v1/dashboard", timeout=5)
            raise AssertionError("served without the token")
        except urllib.error.HTTPError as error:
            assert error.code == 401
    finally:
        server.shutdown()


def test_history_keeps_every_answer_and_marks_yes_runs():
    board = AnswerBoard()
    body = {"camera_id": "cam", "camera_name": "Garage", "questions": {"c1": {"type": "noul", "instructions": "Car?"}},
            "alerts": {"c1": {"on": 0.8}}}
    for i, p in enumerate([0.1, 0.9, 0.95, 0.2, 0.1, 0.9]):
        board.record({**body, "window_end": 1000.0 + 20 * i}, {"c1": {"true": p}}, 2.0, "low", 1.0, now=1001.0 + 20 * i)
    hist = board.history("cam", since=0)
    (q,) = hist["questions"]
    assert [p for _, p in q["points"]] == [0.1, 0.9, 0.95, 0.2, 0.1, 0.9]
    assert q["yes_spans"] == [[990.0, 1040.0], [1070.0, 1100.0]]  # two runs of yes, each from its window start
    assert len(hist["latency"]) == 6


def test_downsampling_keeps_short_yes_spikes():
    from clef_operator.dashboard import downsample
    points = [(i, 0.01) for i in range(1000)]
    points[503] = (503, 0.97)
    thin = downsample(points, 100)
    assert len(thin) == 100 and max(p for _, p in thin) == 0.97


def test_system_history_spreads_model_time_over_minutes_and_keeps_latency_per_class():
    from clef_operator.dashboard import SystemHistory
    sys_h = SystemHistory()
    sys_h.add(110.0, 130.0, "critical", 6.0, True)  # 10 s in minute 1, 10 s in minute 2
    sys_h.add(125.0, 127.0, "low", 25.0, False, lowered=True)
    rows = sys_h.series(since=60.0, now=150.0)
    assert [r["t"] for r in rows] == [60, 120]
    assert rows[0]["busy"] == round(10 / 60, 3) and rows[1]["answers"] == 2 and rows[1]["late"] == 1
    assert rows[1]["latency"]["critical"]["p95"] == 6.0 and rows[1]["lowered"] == 1


def test_dashboard_history_survives_a_restart(tmp_path):
    path = tmp_path / "answers.json"
    board = AnswerBoard(path)
    body = {"camera_id": "cam", "camera_name": "Garage", "questions": {"c1": {"type": "noul", "instructions": "Car?"}}}
    board.record({**body, "window_end": time.time()}, {"c1": {"true": 0.9}}, 2.0, "low", 1.0)
    board.save()
    again = AnswerBoard(path)
    assert again.cameras()[0]["name"] == "Garage" and len(again.history("cam", since=0)["questions"][0]["points"]) == 1
