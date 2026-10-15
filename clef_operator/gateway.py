"""Clef gateway (PRD: Pipeline, part 2): one priority queue in front of the Clef server for every Nx Server.

    POST /v1/jobs       one window's questions for one camera; blocks until answered or dropped
    POST /v1/classify   {"question", "camera"?, "description"?, "interests"?} -> fps, priority, needs_motion
    GET  /v1/status     queue depth, model busy share, done and skipped counts
    POST /v1/hunter     {"camera_id", "enabled", "targets": [...], ...HunterConfig fields} start, update or stop a PTZ hunter
    GET  /v1/hunter     every hunter's state, view and last answers
    GET  /v1/profile?camera_id=...      the camera's learned profile for the current hour
    POST /v1/profile/reset {"camera_id"} forget a camera's learned profile (e.g. after it was moved)
    GET  /health        gateway and Clef server status

A job body:

    {"camera_id": "...", "camera_name": "Front porch", "camera_description": "...",
     "window_start": 1792014600.0, "window_end": 1792014630.0,
     "priority": "critical", "fps": 8, "motion": "onset" | "active" | "still",
     "zone": {"name": "Back door", "description": "...", "box": [x, y, w, h]} | null,
     "site": {"interests": "...", "nuisance": ["flag on the driveway pole", ...]},
     "questions": {"fight": {"type": "noul", "instructions": "..."}},
     "clip": "<base64 MP4>"}

One camera window goes to Clef once per zone. The plugin sends a window's question groups (one per priority and
frame rate) together; the gateway gathers the jobs for one camera and window for ``GATHER_S``, then sends each zone's
clip once, at the highest frame rate and priority asked for, with every question, and gives each caller its own
answers. Clips are resampled, cropped to the zone, and shrunk to ``--max-pixels`` before queuing, in a request
thread, so the model never waits on video work.

    python3 -m clef_operator.gateway --port 8200 --clef http://127.0.0.1:8100
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import signal
import subprocess
import tempfile
import threading
import time
import traceback
from collections import deque
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .classify import classify
from .hunter import Hunter, HunterConfig, NxPtz, View
from .clef import Clef, image_data_url, video_data_url
from .nx import Nx
from .motion import MotionRank
from .profile import SURVEY, ProfileStore, person_probability
from .schedule import TARGET_S, Job, Priority, Scheduler

MAX_BODY = 256 * 1024 * 1024


def log(message: str) -> None:
    print(f"{datetime.now():%H:%M:%S} {message}", flush=True)


def ffmpeg_format(codec: str) -> str:
    """The plugin passes the Server's codec name, e.g. "video/hevc" or "h264"; ffmpeg wants "hevc" or "h264"."""
    name = codec.lower().rsplit("/", 1)[-1]
    if name in ("hevc", "h265", "x-h265", "h.265"):
        return "hevc"
    if name in ("h264", "avc", "x-h264", "h.264"):
        return "h264"
    raise ValueError(f"unsupported codec {codec!r}: the gateway takes H.264 or H.265")


MAX_PIXELS = 448 * 252  # Clef's token floor (~75 tokens a frame): smaller frames cost the same, larger ones more


def prepare_clip(clip: bytes, fps: float, width: int | None = None, box: list[float] | None = None,
                 annexb_codec: str | None = None, source_fps: float | None = None,
                 max_pixels: int | None = MAX_PIXELS) -> bytes:
    """Resample to ``fps``, crop to the normalized ``box`` (plus a 10% margin), and shrink (H.264) to at most
    ``max_pixels`` per frame, or to ``width`` when given.

    ``clip`` is an MP4/MKV, or with ``annexb_codec`` ("h264" or "hevc") a raw elementary stream from the plugin,
    timed at ``source_fps``.
    """
    filters = [f"fps={fps:g}"]
    if box:
        x, y, w, h = box
        mx, my = w * 0.1, h * 0.1
        x0, y0 = max(0.0, x - mx), max(0.0, y - my)
        x1, y1 = min(1.0, x + w + mx), min(1.0, y + h + my)
        filters.append(f"crop=iw*{x1 - x0:.4f}:ih*{y1 - y0:.4f}:iw*{x0:.4f}:ih*{y0:.4f}")
    if width:
        filters.append(f"scale='min({width},iw)':-2")
    elif max_pixels:
        filters.append(f"scale='trunc(min(iw,iw*sqrt({max_pixels}/(iw*ih)))/2)*2':-2")
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = Path(tmp) / "in.mp4", Path(tmp) / "out.mp4"
        src.write_bytes(clip)
        source = ["-i", str(src)]
        if annexb_codec:
            source = ["-f", ffmpeg_format(annexb_codec),
                      "-r", f"{source_fps or 25:.3f}", "-i", str(src)]
        run = subprocess.run(
            ["ffmpeg", "-nostdin", "-loglevel", "error", *source, "-vf", ",".join(filters),
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-an", "-y", str(dst)],
            capture_output=True, text=True, timeout=120,
        )
        if run.returncode != 0:
            keep = Path(tempfile.gettempdir()) / f"clefop_failed_{int(time.time())}.{ffmpeg_format(annexb_codec) if annexb_codec else 'mp4'}"
            keep.write_bytes(clip)
            raise ValueError(f"ffmpeg exit {run.returncode}: {run.stderr.strip()[-400:]} (input kept at {keep})")
        return dst.read_bytes()


def prepare_image(jpeg: bytes, max_pixels: int | None = MAX_PIXELS) -> bytes:
    """Shrink a snapshot to Clef's token floor (snapshots arrive at full camera resolution)."""
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = Path(tmp) / "in.jpg", Path(tmp) / "out.jpg"
        src.write_bytes(jpeg)
        subprocess.run(
            ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(src),
             "-vf", f"scale='trunc(min(iw,iw*sqrt({max_pixels}/(iw*ih)))/2)*2':-2", "-q:v", "3", "-y", str(dst)],
            check=True, timeout=60,
        )
        return dst.read_bytes()


def build_state(job: dict, now: float) -> str:
    parts = [f"Security camera '{job.get('camera_name') or job['camera_id']}'."]
    if job.get("camera_description"):
        parts.append(f"About this camera: {job['camera_description']}")
    if job.get("profile"):
        parts.append(f"What it usually sees at this time: {job['profile']}")
    zone = job.get("zone")
    if zone:
        parts.append(f"This clip shows only the zone '{zone['name']}'." + (f" {zone['description']}" if zone.get("description") else ""))
    site = job.get("site") or {}
    if site.get("interests"):
        parts.append(f"What matters at this site: {site['interests']}")
    if site.get("nuisance"):
        parts.append("Known harmless movement to ignore: " + "; ".join(site["nuisance"]) + ".")
    window_end = float(job.get("window_end", now))
    parts.append(f"Clip ends {datetime.fromtimestamp(window_end):%A %H:%M:%S} local time.")
    return " ".join(parts)


class Pending:
    """A queued job's request thread waits on this until the worker answers or drops the job."""

    def __init__(self) -> None:
        self.done = threading.Event()
        self.result: dict | None = None


GATHER_S = 0.3  # the plugin sends a window's question groups at once: wait this long for the rest of them


class Part:
    """One caller's job body inside a bundled Clef request, and how its question ids were sent."""

    def __init__(self, body: dict, arrived: float = 0.0) -> None:
        self.body = body
        self.arrived = arrived or time.time()
        self.pending = Pending()
        self.ids: dict[str, str] = {}  # id sent to Clef -> the caller's question id
        self.error: Exception | None = None


def bundle_questions(parts: list[Part]) -> dict[str, dict]:
    """Every part's questions in one request; an id two parts share is renamed for Clef and mapped back."""
    questions: dict[str, dict] = {}
    for i, part in enumerate(parts):
        part.ids = {}
        for qid, question in part.body["questions"].items():
            sent = qid if qid not in questions else f"{qid}~{i}"
            questions[sent] = question
            part.ids[sent] = qid
    return questions


def camera_label(job: Job) -> str:
    parts = job.payload.get("parts")
    name = (parts[0].body.get("camera_name") if parts else job.payload.get("camera_name")) or job.camera_id.strip("{}")[:8]
    return f"{name} (PTZ hunter)" if job.group == "hunter" else name


def latency_by_camera(records) -> dict:
    """Per camera: answers, mean, 95th percentile and worst seconds from arrival, and the share on time."""
    rows: dict[str, list] = {}
    for record in records:
        rows.setdefault(record[5], []).append((record[2], record[3]))
    out = {}
    for name, values in sorted(rows.items()):
        times = sorted(v[0] for v in values)
        out[name] = {"n": len(times), "mean": round(sum(times) / len(times), 2),
                     "p95": round(times[min(len(times) - 1, int(len(times) * 0.95))], 2), "max": round(times[-1], 2),
                     "on_time": round(sum(v[1] for v in values) / len(values), 3)}
    return out


def latency_summary(records) -> dict:
    """Per class: answers, median and 95th percentile seconds from arrival, worst, and the share on time."""
    by_class: dict[str, list] = {}
    for _, name, seconds, on_time, lowered, *_ in records:
        by_class.setdefault(name, []).append((seconds, on_time, lowered))
    out = {}
    for name, rows in by_class.items():
        times = sorted(r[0] for r in rows)
        out[name.lower()] = {"n": len(rows), "p50": round(times[len(times) // 2], 2),
                             "p95": round(times[min(len(times) - 1, int(len(times) * 0.95))], 2),
                             "max": round(times[-1], 2), "on_time": round(sum(r[1] for r in rows) / len(rows), 3),
                             "lowered": sum(r[2] for r in rows),
                             "target_s": TARGET_S[Priority[name]]}
    return out


def finish(job: Job, result: dict) -> None:
    """Hand a job's result to its caller, or to each bundled caller with only its own answers."""
    parts: list[Part] | None = job.payload.get("parts")
    if parts is None:
        job.payload["pending"].result = result
        job.payload["pending"].done.set()
        return
    for part in parts:
        own = dict(result)
        if "answers" in result:
            own["answers"] = {qid: result["answers"][sent] for sent, qid in part.ids.items() if sent in result["answers"]}
            own["bundled"] = len(parts)
        part.pending.result = own
        part.pending.done.set()


class Gateway:
    def __init__(self, clef: Clef, max_pixels: int, profiles: ProfileStore | None = None,
                 state_dir: Path | None = None) -> None:
        self.clef, self.max_pixels = clef, max_pixels
        self.state_dir = state_dir
        self.presets_path = (state_dir / "ptz_presets.json") if state_dir else None
        self.profiles = profiles or ProfileStore()
        self.boost_until: dict[str, float] = {}  # camera -> time; unusual activity puts its jobs first in class
        self.hunters: dict[str, Hunter] = {}
        self._nx: Nx | None = None
        self.scheduler = Scheduler()
        self.wake = threading.Condition()
        self.bundles: dict[tuple, list[Part]] = {}  # (camera, window end) -> parts still gathering
        self.bundle_lock = threading.Lock()
        self.busy: deque[tuple[float, float]] = deque()  # (start, end) of model requests, last hour
        self.latency: deque[tuple] = deque()  # (answered at, class, seconds from arrival, on time, lowered, camera), last hour
        self.started = time.time()
        threading.Thread(target=self._work, daemon=True, name="worker").start()
        threading.Thread(target=self._sweep, daemon=True, name="sweeper").start()

    # The worker: one Clef request at a time, always the best job next.
    def _work(self) -> None:
        while True:
            with self.wake:
                job = self.scheduler.next(time.time())
                while job is None:
                    self.wake.wait(timeout=1.0)
                    job = self.scheduler.next(time.time())
            started = time.time()
            result: dict = {"status": "error", "error": "not run"}
            try:
                if "images" in job.payload:
                    result = self.clef.ask(job.payload["state"], job.payload["questions"], images=job.payload["images"])
                else:
                    video = job.payload["video"]
                    if job.fps < job.payload.get("clip_fps", job.fps):  # lowered to keep the queue on time
                        video = video_data_url(prepare_clip(job.payload["clip"], job.fps, None, None, max_pixels=None))
                    result = self.clef.ask(job.payload["state"], job.payload["questions"],
                                           videos=[video], video_fps=job.fps)
                self.scheduler.cost.learn(result.input_tokens, result.seconds)
                if job.payload.get("survey"):
                    survey = {q: result.answers.pop(q) for q in SURVEY}
                    self.profiles.observe(job.camera_id, job.window_end, survey)
                    if self.profiles.unusual(job.camera_id, job.window_end, person_probability(survey)):
                        self.boost_until[job.camera_id] = time.time() + 120
                        log(f"{job.camera_id[:8]} unusual for this hour: boosting its windows for 2 minutes")
                self._keep_for_review(job, result.answers)
                hunter = self.hunters.get(job.camera_id.strip("{}")) if job.group != "hunter" else None
                start, end = job.payload.get("window", (None, None))
                yes = [a["true"] for a in result.answers.values() if "true" in a]
                if hunter and start is not None and end is not None and yes:
                    hunter.observe_window(float(start), float(end), max(yes))  # what the camera's own questions saw
                answered = result
                result = {
                    "status": "answered", "answers": answered.answers, "tokens": answered.input_tokens,
                    "model_seconds": round(answered.seconds, 3), "queue_seconds": round(started - job.queued_at, 3),
                    "fps": job.fps, "priority": job.priority.name.lower(),
                    "latency_seconds": round(time.time() - job.arrived, 3) if job.arrived else None,
                }
                shown = " ".join(f"{q}={a['true']:.2f}" if "true" in a else f"{q}={max(a, key=a.get)}"
                                 for q, a in answered.answers.items())
                callers = len(job.payload.get("parts", ())) or 1
                group = job.group + (f"+{callers - 1}" if callers > 1 else "")
                lowered = f" (from {job.requested_fps:g} fps)" if job.fps < job.requested_fps else ""
                log(f"{job.camera_id[:8]} {job.priority.name:<11} {group:<24} {answered.input_tokens:>6} tok "
                    f"{answered.seconds:5.2f}s queued {started - job.queued_at:5.2f}s{lowered}  {shown}")
            except Exception as error:  # noqa: BLE001 - report to the caller, keep the worker alive
                result = {"status": "error", "error": str(error)}
                log(f"{job.camera_id[:8]} error {error!r}")
            finally:
                ended = time.time()
                self.busy.append((started, ended))
                if result.get("status") == "answered" and job.arrived:
                    self.latency.append((ended, job.priority.name, ended - job.arrived, ended <= job.due(),
                                         job.fps < job.requested_fps, camera_label(job)))
                finish(job, result)

    # Dropped jobs (expired or merged) never reach the worker; release their callers.
    def _sweep(self) -> None:
        while True:
            time.sleep(0.5)
            for job in self.scheduler.take_dropped():
                finish(job, {"status": "dropped", "reason": job.drop_reason})

    def submit(self, body: dict) -> dict:
        """Queue one window's questions and wait for the answers. Jobs for the same camera and window that arrive
        within ``GATHER_S`` of each other share Clef requests: one per zone."""
        now = time.time()
        hunter = self.hunters.get(str(body.get("camera_id", "")).strip("{}"))
        if hunter:
            hunter.window_arrived()
        if hunter and hunter.moved_during(float(body.get("window_start", now)), float(body.get("window_end", now))):
            return {"status": "dropped", "reason": "camera moving (PTZ hunter)"}
        part = Part(body, now)
        key = (body["camera_id"], round(float(body["window_end"]), 1)) if "window_end" in body else None
        leader = True
        if key is not None:
            with self.bundle_lock:
                if key in self.bundles:
                    self.bundles[key].append(part)
                    leader = False
                else:
                    self.bundles[key] = [part]
        if leader:
            parts = [part]
            if key is not None:
                time.sleep(GATHER_S)
                with self.bundle_lock:
                    parts = self.bundles.pop(key)
            zones: dict[tuple, list[Part]] = {}
            for p in parts:
                zone = p.body.get("zone") or {}
                zones.setdefault((zone.get("name", ""), tuple(zone.get("box") or ())), []).append(p)
            for zone_parts in zones.values():
                try:
                    self._enqueue(zone_parts, now)
                except Exception as error:  # noqa: BLE001 - every caller gets the error it would have had alone
                    for p in zone_parts:
                        p.error = error
                        p.pending.done.set()
        part.pending.done.wait(timeout=float(body.get("timeout", 600)))
        if part.error is not None:
            raise part.error
        return part.pending.result or {"status": "dropped", "reason": "timeout"}

    def _enqueue(self, parts: list[Part], now: float) -> None:
        """One Clef request for ``parts`` (one camera, window and zone): the highest frame rate, priority and motion
        any of them asked for, and all their questions."""
        first = parts[0].body
        camera_id = first["camera_id"]
        fps = max(float(p.body.get("fps", 2)) for p in parts)
        priority = min(Priority[p.body.get("priority", "normal").upper()] for p in parts)
        motion = min(MotionRank[p.body.get("motion", "active").upper()] for p in parts)
        if now < self.boost_until.get(camera_id, 0.0):
            motion = MotionRank.ONSET
        annexb = first.get("codec") if first.get("clip_format") == "annexb" else None
        clip = prepare_clip(base64.b64decode(first["clip"]), fps, None, (first.get("zone") or {}).get("box"),
                            annexb, first.get("clip_fps"), self.max_pixels)
        first["profile"] = self.profiles.text(camera_id, float(first.get("window_end", now)))
        questions = bundle_questions(parts)
        survey = self.profiles.survey_due(camera_id, now)
        if survey:
            questions.update(SURVEY)  # the video is going anyway; the survey adds only its text
        start, end = first.get("window_start"), first.get("window_end")
        job = Job(
            arrived=min(p.arrived for p in parts),
            seconds=float(end) - float(start) if start is not None and end is not None else 30.0,
            camera_id=camera_id,
            # The priority is part of the group: merging replaces older Normal/Low windows of the same group, and a
            # Critical question at the same fps must never be merged away by a newer Normal one.
            group=first.get("group") or f"{priority.name.lower()}|fps={fps:g}|zone={(first.get('zone') or {}).get('name', '')}",
            priority=priority,
            window_end=float(first.get("window_end", now)),
            motion=motion,
            fps=fps,
            payload={"parts": parts, "questions": questions, "state": build_state(first, now),
                     "video": video_data_url(clip), "survey": survey, "clip": clip, "clip_fps": fps,
                     "window": (first.get("window_start"), first.get("window_end"))},
        )
        with self.wake:
            self.scheduler.add(job, now)
            self.wake.notify()

    def _keep_for_review(self, job: Job, answers: dict, keep: int = 60) -> None:
        """The last ``keep`` clips per camera, exactly as Clef saw them, with its answers: so a missed car can be
        checked frame by frame even when the camera only records on objects."""
        if not self.state_dir or "clip" not in job.payload:
            return
        try:
            folder = self.state_dir / "review" / job.camera_id.strip("{}")
            folder.mkdir(parents=True, exist_ok=True)
            start, end = job.payload.get("window") or (None, None)
            stamp = datetime.fromtimestamp(float(end or time.time())).strftime("%Y%m%d-%H%M%S")
            name = f"{stamp}_{job.priority.name.lower()}"
            (folder / f"{name}.mp4").write_bytes(job.payload["clip"])
            (folder / f"{name}.json").write_text(json.dumps({
                "window_start": start, "window_end": end, "priority": job.priority.name, "fps": job.fps,
                "motion": job.motion.name, "group": job.group,
                "answers": {q: a.get("true", a) for q, a in answers.items()}}))
            clips = sorted(folder.glob("*.mp4"))
            for old in clips[:-keep]:
                old.unlink(missing_ok=True)
                old.with_suffix(".json").unlink(missing_ok=True)
        except Exception as error:  # noqa: BLE001 - review copies must never break answering
            log(f"review store failed: {error!r}")

    @property
    def nx(self) -> Nx:
        if self._nx is None:
            self._nx = Nx()
        return self._nx

    def ask_image(self, camera_id: str, jpeg: bytes | list[bytes], questions: dict, state: str = "",
                  priority: str = "low") -> dict:
        """One snapshot (or several images in one request) through the queue (Low by default: hunting uses spare
        GPU time)."""
        images = [image_data_url(prepare_image(j, self.max_pixels)) for j in (jpeg if isinstance(jpeg, list) else [jpeg])]
        pending = Pending()
        now = time.time()
        hunter = self.hunters.get(camera_id.strip("{}"))
        job = Job(camera_id=camera_id, group="hunter", priority=Priority[priority.upper()], window_end=now,
                  motion=MotionRank.ACTIVE, fps=1.0, arrived=now, images=len(images),
                  payload={"pending": pending, "questions": questions, "images": images,
                           "camera_name": hunter.config.camera_name.strip() if hunter else "",
                           "state": state or f"A snapshot from a PTZ security camera. Local time {datetime.now():%A %H:%M}."})
        with self.wake:
            self.scheduler.add(job, now)
            self.wake.notify()
        pending.done.wait(timeout=300)
        result = pending.result or {"status": "dropped", "reason": "timeout"}
        if result["status"] != "answered":
            raise RuntimeError(f"hunter question not answered: {result}")
        return result["answers"]

    def set_hunter(self, body: dict) -> dict:
        camera_id = body["camera_id"].strip("{}")
        previous = self.hunters.get(camera_id)
        for key in ("home_preset", "tilt_min", "tilt_max"):  # the plugin does not send these yet: keep the ones set here
            if key not in body:
                saved = getattr(previous.config, key) if previous else self._saved_hunter_field(camera_id, key)
                if saved not in (None, ""):
                    body = {**body, key: saved}
        fields = {k: v for k, v in body.items() if k in HunterConfig.__dataclass_fields__ and k != "camera_id"}
        config = HunterConfig(camera_id=camera_id, **fields)
        running = self.hunters.get(camera_id)
        if running and body.get("enabled", True) and running.config == config and running.status.state not in ("stopped", "error"):
            return {"status": running.status.state}  # saving unchanged settings must not restart the survey
        if body.get("enabled", True) and self.nx.ptz_operational(camera_id) is False:
            if running and running.status.state not in ("stopped", "error"):
                # E.g. the Server is restarting and re-sent the settings before its PTZ was ready: keep hunting.
                log(f"hunter {camera_id[:8]}: the Server cannot drive its PTZ right now; keeping the running hunter")
                return {"status": running.status.state, "warning": "the Server cannot drive this PTZ right now"}
            return {"status": "error", "error": "not a PTZ camera the Nx Server can drive"}
        old = self.hunters.pop(camera_id, None)
        if not body.get("enabled", True):
            if old:
                old.stop(wait=False, return_home=True)  # turned off: drive home slowly
            self._save_hunter_configs()
            return {"status": "stopped" if old else "off"}
        if old:
            old.stop(wait=True, return_home=False)  # replaced: stop the motor here; the new hunter takes over
        # None (could not tell yet, e.g. the Server is starting): the hunter waits for the PTZ before it moves.
        state = (f"A snapshot from the PTZ security camera '{config.camera_name}'. {config.description} "
                 f"The camera is searching for a better view.").strip()

        def ask(cam: str, jpeg: bytes, questions: dict) -> dict:
            return self.ask_image(cam, jpeg, questions, state)

        def found(cfg: HunterConfig, view: View, question: str, p: float) -> None:
            log(f"hunter {cfg.camera_id[:8]} FOUND '{question}' p={p:.2f} at pan {view.pan} tilt {view.tilt} fov {view.fov}")
            try:
                self.nx.generic_event("Clef PTZ hunter", f"{cfg.camera_name}: {question}",
                                      f"Yes · {p:.2f} · pan {view.pan:.1f}° tilt {view.tilt:+.1f}° field of view {view.fov:.1f}°",
                                      [cfg.camera_id])
            except Exception as error:  # noqa: BLE001 - the dwell matters more than the event
                log(f"hunter generic event failed: {error!r}")

        memory = (self.state_dir / f"hunter_{camera_id}.json") if self.state_dir else None
        hunter = Hunter(config, NxPtz(self.nx, camera_id), ask, on_found=found,
                        locate_preset=lambda preset: self.locate_preset(camera_id, preset), memory=memory)
        self.hunters[camera_id] = hunter
        hunter.start()
        self._save_hunter_configs()
        log(f"hunter {camera_id[:8]} started: {config.targets}, max speed {config.max_speed}, rest {config.rest_s}s")
        return {"status": "started"}

    def _save_hunter_configs(self) -> None:
        """Running hunters are restarted when the gateway restarts (the plugin only resends on a settings change)."""
        if self.state_dir:
            configs = [vars(h.config) for h in self.hunters.values()]
            (self.state_dir / "hunters.json").write_text(json.dumps(configs, indent=1))

    def _saved_hunter_field(self, camera_id: str, key: str):
        path = self.state_dir / "hunters.json" if self.state_dir else None
        if path and path.exists():
            for config in json.loads(path.read_text()):
                if config.get("camera_id") == camera_id:
                    return config.get(key)
        return None

    def restore_hunters(self) -> None:
        path = self.state_dir / "hunters.json" if self.state_dir else None
        if path and path.exists():
            for config in json.loads(path.read_text()):
                log(f"restoring hunter {config['camera_id'][:8]}: {self.set_hunter({**config, 'enabled': True})}")

    def locate_preset(self, camera_id: str, preset: str) -> View | None:
        """Where a preset points, in logical degrees. Learned once by visiting it (a normal-speed preset move, the
        only fast move the hunter ever makes), then cached in the state directory."""
        cache = json.loads(self.presets_path.read_text()) if self.presets_path and self.presets_path.exists() else {}
        known = cache.get(camera_id, {}).get(str(preset))
        if known:
            return View(known["pan"], known["tilt"], known["fov"])
        ptz = NxPtz(self.nx, camera_id)
        try:
            self.nx._request("POST", f"/rest/v4/devices/{camera_id}/ptz/presets/{preset}/activate", {"speed": 0.3})
        except Exception as error:  # noqa: BLE001
            log(f"hunter {camera_id[:8]} cannot activate preset {preset}: {error!r}")
            return None
        last, deadline = None, time.time() + 30
        while time.time() < deadline:  # wait for the camera to stop
            time.sleep(1.0)
            now = ptz.position()
            if last and now.distance(last) < 0.2:
                break
            last = now
        cache.setdefault(camera_id, {})[str(preset)] = {"pan": now.pan, "tilt": now.tilt, "fov": now.fov}
        if self.presets_path:
            self.presets_path.write_text(json.dumps(cache, indent=1))
        log(f"hunter {camera_id[:8]} learned preset {preset}: pan {now.pan:.1f} tilt {now.tilt:.1f} fov {now.fov:.1f}")
        return now

    def hunter_status(self) -> dict:
        out = {}
        for camera_id, h in self.hunters.items():
            st = h.status
            learned = h.learned(datetime.now().hour)
            out[camera_id] = {
                "state": st.state, "view": vars(st.view) if st.view else None, "found": st.found, "error": st.error,
                "moves": st.moves, "last_answers": st.last_answers,
                "hot_spots": [{"view": vars(v), "finds": s.total_hits, "visits": s.visits} for v, s in h.hot_spots()],
                "best_views": [r for r in learned if not r["quiet"]][:6],
                "quiet_views": [r["view"] for r in learned if r["quiet"]],
            }
        return out

    def status(self) -> dict:
        now = time.time()
        while self.busy and self.busy[0][1] < now - 3600:
            self.busy.popleft()
        def share(seconds: float) -> float:
            busy = sum(min(end, now) - max(start, now - seconds) for start, end in self.busy if end > now - seconds)
            return round(busy / min(seconds, now - self.started), 3) if now > self.started else 0.0
        while self.latency and self.latency[0][0] < now - 3600:
            self.latency.popleft()
        return {"queue": self.scheduler.depth(), "busy_1m": share(60), "busy_10m": share(600), "busy_1h": share(3600),
                "done": dict(self.scheduler.done), "skipped": dict(self.scheduler.skipped),
                "lowered_fps": dict(self.scheduler.lowered), "latency_1h": latency_summary(self.latency),
                "latency_by_camera_1h": latency_by_camera(self.latency),
                "seconds_per_1k_tokens": round(self.scheduler.cost.per_token_s * 1000, 3)}


def make_handler(gateway: Gateway, token: str | None):
    class Handler(BaseHTTPRequestHandler):
        server_version = "clef-gateway/0.1"

        def _send(self, code: int, body: dict) -> None:
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _body(self) -> dict:
            length = int(self.headers.get("Content-Length", 0))
            if not 0 < length <= MAX_BODY:
                raise ValueError("a JSON body is required")
            body = json.loads(self.rfile.read(length))
            if not isinstance(body, dict):
                raise ValueError("the body must be a JSON object")
            return body

        def _authorized(self) -> bool:
            return not token or self.headers.get("Authorization", "") == f"Bearer {token}"

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                try:
                    clef = gateway.clef.health()
                except Exception as error:  # noqa: BLE001
                    clef = {"status": "unreachable", "error": str(error)}
                return self._send(200, {"status": "ready" if clef.get("status") == "ready" else "degraded", "clef": clef})
            if not self._authorized():
                return self._send(401, {"error": "missing or wrong bearer token"})
            if self.path == "/v1/status":
                return self._send(200, gateway.status())
            if self.path == "/v1/hunter":
                return self._send(200, gateway.hunter_status())
            if self.path.startswith("/v1/profile?camera_id="):
                camera = self.path.split("=", 1)[1]
                return self._send(200, {"camera_id": camera, "profile": gateway.profiles.text(camera, time.time())})
            self._send(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            if not self._authorized():
                return self._send(401, {"error": "missing or wrong bearer token"})
            try:
                body = self._body()
                if self.path == "/v1/jobs":
                    return self._send(200, gateway.submit(body))
                if self.path == "/v1/hunter":
                    return self._send(200, gateway.set_hunter(body))
                if self.path == "/v1/profile/reset":
                    gateway.profiles.reset(body["camera_id"])
                    return self._send(200, {"status": "reset"})
                if self.path == "/v1/classify":
                    c = classify(gateway.clef, body["question"], body.get("camera", ""), body.get("description", ""),
                                 body.get("interests", ""))
                    return self._send(200, {"fps": c.fps, "priority": c.priority.name.lower(), "needs_motion": c.needs_motion,
                                            "summary": c.summary(), "fps_probabilities": c.fps_probabilities,
                                            "priority_probabilities": c.priority_probabilities,
                                            "needs_motion_probability": c.needs_motion_probability})
                self._send(404, {"error": "not found"})
            except (ValueError, KeyError, TypeError, json.JSONDecodeError, subprocess.CalledProcessError) as error:
                self._send(400, {"error": repr(error)})
            except Exception as error:  # noqa: BLE001
                log(f"500 {error!r}\n{traceback.format_exc()}")
                self._send(500, {"error": repr(error)})

        def log_message(self, format: str, *args) -> None:  # the worker logs each job
            pass

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8200)
    parser.add_argument("--clef", default=None, help="Clef server URL (default: CLEF_URL)")
    parser.add_argument("--max-pixels", type=int, default=MAX_PIXELS, help="frames are shrunk to at most this area")
    parser.add_argument("--state-dir", type=Path, default=Path.home() / ".clef_operator", help="learned profiles live here")
    args = parser.parse_args()
    args.state_dir.mkdir(parents=True, exist_ok=True)
    gateway = Gateway(Clef(args.clef), args.max_pixels, ProfileStore(args.state_dir / "profiles.json"), args.state_dir)
    threading.Thread(target=gateway.restore_hunters, daemon=True).start()

    def shutdown(signum, frame) -> None:
        # A hunter killed mid-move would leave the PTZ motor running in continuous mode: stop every motor first.
        # The hunter configs stay saved, so the next start resumes them.
        for camera_id, hunter in list(gateway.hunters.items()):
            hunter._stop.set()
            try:
                hunter.ptz.continuous(0.0, 0.0, 0.0)
                log(f"hunter {camera_id[:8]}: motor stopped for shutdown")
            except Exception as error:  # noqa: BLE001
                log(f"hunter {camera_id[:8]}: could not stop motor: {error!r}")
        os._exit(0)

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    token = os.environ.get("CLEF_GATEWAY_TOKEN")
    log(f"gateway on {args.host}:{args.port} -> {gateway.clef.url}, max {args.max_pixels} px, auth {'on' if token else 'off'}")
    ThreadingHTTPServer((args.host, args.port), make_handler(gateway, token)).serve_forever()


if __name__ == "__main__":
    main()
