"""The integration's dashboard: answers per camera over time, Clef's load and the PTZ hunters.

It replaces the answers drawn on the video (one analytics object per camera), which made cameras that record on
objects record all the time. The gateway serves the page (``static/dashboard.html``) at ``/`` and its data under
``/v1/``; Nx shows the page as a Web Page under Integrations. Nothing in it names a camera: cameras come from Nx and
from the answers themselves.

History is kept for ``HISTORY_S`` (one point per answered window per question, and one row per minute for the
system) and saved to the gateway's state directory, so a restart keeps the graphs.
"""

from __future__ import annotations

import json
import math
import threading
import time
from collections import deque
from pathlib import Path

from .events import Question

HISTORY_S = 24 * 3600.0
PAGE_PATH = Path(__file__).with_name("static") / "dashboard.html"
STRIP_PATH = Path(__file__).with_name("static") / "strip.html"


def page() -> str:
    return PAGE_PATH.read_text()


def strip_page() -> str:
    """One camera's answers and their last hour, small enough to sit beside the camera in an Nx layout
    (``/strip?camera=<id>``; without a camera, a list to pick one)."""
    return STRIP_PATH.read_text()


class AnswerBoard:
    """The latest answer to each (camera, question), and every answer of the last ``HISTORY_S``."""

    def __init__(self, path: Path | None = None) -> None:
        self._lock = threading.Lock()
        self._cameras: dict[str, dict] = {}
        self._history: dict[tuple[str, str], deque] = {}  # (camera, question) -> (t, p, yes)
        self._latency: dict[str, deque] = {}  # camera -> (t, seconds)
        self.path = path
        if path and path.exists():
            self._load(json.loads(path.read_text()))

    def record(self, body: dict, answers: dict[str, dict], fps: float, priority: str, latency_s: float | None,
               now: float | None = None) -> None:
        """``answers`` is keyed by the caller's question ids (only this caller's questions)."""
        now = time.time() if now is None else now
        camera_id = str(body["camera_id"]).strip("{}")
        zone = (body.get("zone") or {}).get("name", "")
        alerts = body.get("alerts") or {}
        window_end = float(body.get("window_end", now))
        rows = {}
        for qid, answer in answers.items():
            spec = body["questions"].get(qid)
            if spec is not None:
                rows[qid] = answer_row(qid, spec, answer, alerts.get(qid, {}), zone, fps, priority, window_end, now,
                                       latency_s)
        with self._lock:
            camera = self._cameras.setdefault(camera_id, {"id": camera_id, "questions": {}})
            camera["name"] = body.get("camera_name") or camera.get("name") or camera_id[:8]
            camera["questions"].update(rows)
            camera["answered_at"] = now
            for qid, row in rows.items():
                self._history.setdefault((camera_id, qid), deque()).append((window_end, row["p"], row["yes"]))
            if latency_s is not None:
                self._latency.setdefault(camera_id, deque()).append((now, round(latency_s, 2)))
            self._trim(now)

    def _trim(self, now: float) -> None:
        cutoff = now - HISTORY_S
        for series in list(self._history.values()) + list(self._latency.values()):
            while series and series[0][0] < cutoff:
                series.popleft()

    def cameras(self, now: float | None = None, only: set[str] | None = None) -> list[dict]:
        now = time.time() if now is None else now
        with self._lock:
            out = []
            for camera in sorted(self._cameras.values(), key=lambda c: c["name"].lower()):
                if only is not None and camera["id"] not in only:
                    continue
                questions = sorted(camera["questions"].values(), key=lambda q: (q["zone"], q["id"]))
                out.append({"id": camera["id"], "name": camera["name"], "age_s": round(now - camera["answered_at"], 1),
                            "questions": [{**q, "age_s": round(now - q["answered_at"], 1)} for q in questions]})
            return out

    def history(self, camera_id: str, since: float, points: int = 400) -> dict:
        """Each question's probability over time (downsampled to about ``points``, keeping peaks), its yes spans, and
        the time each answer took."""
        camera_id = camera_id.strip("{}")
        with self._lock:
            camera = self._cameras.get(camera_id)
            if camera is None:
                return {"id": camera_id, "questions": [], "latency": []}
            questions = []
            for qid, row in sorted(camera["questions"].items(), key=lambda kv: (kv[1]["zone"], kv[0])):
                series = [s for s in self._history.get((camera_id, qid), ()) if s[0] >= since]
                questions.append({"id": qid, "text": row["text"], "zone": row["zone"], "on": row["on"],
                                  "kind": row["kind"], "points": downsample([(t, p) for t, p, _ in series], points),
                                  "yes_spans": yes_spans(series)})
            latency = downsample([s for s in self._latency.get(camera_id, ()) if s[0] >= since], points)
            return {"id": camera_id, "name": camera["name"], "questions": questions, "latency": latency}

    def save(self) -> None:
        if not self.path:
            return
        with self._lock:
            data = {"cameras": self._cameras,
                    "history": [[c, q, list(s)] for (c, q), s in self._history.items()],
                    "latency": {c: list(s) for c, s in self._latency.items()}}
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data))
        tmp.replace(self.path)

    def _load(self, data: dict) -> None:
        self._cameras = data.get("cameras", {})
        self._history = {(c, q): deque(tuple(p) for p in s) for c, q, s in data.get("history", [])}
        self._latency = {c: deque(tuple(p) for p in s) for c, s in data.get("latency", {}).items()}
        self._trim(time.time())


class SystemHistory:
    """One row per minute: Clef busy seconds, answers, and time to answer per class, for ``HISTORY_S``."""

    def __init__(self, path: Path | None = None) -> None:
        self._lock = threading.Lock()
        self._minutes: dict[int, dict] = {}
        self.path = path
        if path and path.exists():
            self._minutes = {int(k): v for k, v in json.loads(path.read_text()).items()}

    def add(self, start: float, end: float, priority: str | None = None, latency_s: float | None = None,
            on_time: bool | None = None, lowered: bool = False) -> None:
        with self._lock:
            t = start
            while t < end:  # spread the model time over the minutes it covers
                minute = int(t // 60)
                step = min(end, (minute + 1) * 60) - t
                self._row(minute)["busy_s"] += step
                t += step
            if priority is not None and latency_s is not None:
                row = self._row(int(end // 60))
                row["answers"] += 1
                row["latency"].setdefault(priority, []).append(round(latency_s, 2))
                row["late"] += 0 if on_time else 1
                row["lowered"] += 1 if lowered else 0
            cutoff = int((end - HISTORY_S) // 60)
            for minute in [m for m in self._minutes if m < cutoff]:
                del self._minutes[minute]

    def _row(self, minute: int) -> dict:
        return self._minutes.setdefault(minute, {"busy_s": 0.0, "answers": 0, "late": 0, "lowered": 0, "latency": {}})

    def series(self, since: float, now: float | None = None) -> list[dict]:
        """Per minute from ``since`` (or the first minute recorded, if later): busy share, answers, late, lowered, and
        median and 95th-percentile time to answer per class (quiet minutes are included, so gaps show as zero load)."""
        now = time.time() if now is None else now
        with self._lock:
            rows = []
            known = [m for m in self._minutes if m >= int(since // 60)]
            first = min(known) if known else int(now // 60)  # before the first record there is nothing to show
            for minute in range(first, int(now // 60) + 1):
                row = self._minutes.get(minute)
                entry = {"t": minute * 60, "busy": 0.0, "answers": 0, "late": 0, "lowered": 0, "latency": {}}
                if row:
                    entry.update(busy=round(min(1.0, row["busy_s"] / 60.0), 3), answers=row["answers"],
                                 late=row["late"], lowered=row["lowered"])
                    for cls, values in row["latency"].items():
                        ordered = sorted(values)
                        entry["latency"][cls] = {"p50": ordered[len(ordered) // 2],
                                                 "p95": ordered[min(len(ordered) - 1, math.ceil(len(ordered) * 0.95) - 1)]}
                rows.append(entry)
            return rows

    def save(self) -> None:
        if not self.path:
            return
        with self._lock:
            data = json.dumps(self._minutes)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(data)
        tmp.replace(self.path)


def downsample(points: list[tuple], limit: int) -> list[list]:
    """At most ``limit`` points: per bucket, the point with the highest value, so short yes spikes survive."""
    if len(points) <= limit:
        return [list(p) for p in points]
    size = len(points) / limit
    out = []
    for i in range(limit):
        bucket = points[int(i * size): max(int(i * size) + 1, int((i + 1) * size))]
        out.append(list(max(bucket, key=lambda p: p[1])))
    return out


def yes_spans(series: list[tuple], gap_s: float = 45.0) -> list[list[float]]:
    """[start, end] of runs of yes answers (windows are 20 s apart; a longer gap ends a run)."""
    spans: list[list[float]] = []
    for t, _, yes in series:
        if not yes:
            continue
        if spans and t - spans[-1][1] <= gap_s:
            spans[-1][1] = t
        else:
            spans.append([t - 30.0, t])  # a window covers the 30 s before its end
    return spans


def answer_row(qid: str, spec: dict, answer: dict[str, float], alert: dict, zone: str, fps: float, priority: str,
               window_end: float, now: float, latency_s: float | None) -> dict:
    """One question's latest answer, judged the way the plugin judges it (its on-threshold and alerting options)."""
    kind = spec.get("type", "noul")
    criteria = spec.get("criteria") or []
    options = list(criteria) if isinstance(criteria, (dict, list)) else []
    question = Question(qid, spec.get("instructions", qid), kind, options, list(alert.get("alert_on") or []),
                        float(alert.get("on", 0.8)))
    try:
        p = question.alert_probability(answer)
    except (KeyError, IndexError, ValueError):
        p = 0.0
    shown = max(answer, key=answer.get) if answer and kind != "noul" else None
    return {"id": qid, "text": question.text, "kind": kind, "zone": zone, "p": round(p, 3),
            "yes": p >= question.on_threshold, "on": question.on_threshold, "option": shown,
            "option_p": round(answer.get(shown, 0.0), 3) if shown else None, "priority": priority, "fps": fps,
            "window_end": window_end, "answered_at": now,
            "latency_s": round(latency_s, 2) if latency_s is not None else None}
