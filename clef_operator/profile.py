"""Learned camera profiles (PRD: Camera knowledge).

Once an hour per camera, the gateway adds a few survey questions to a request that is already going to Clef (the
video is sent anyway, so the survey costs only its text). The answers are tallied per hour of day and weekday or
weekend, with older days weighted down, and a template turns the current slot's tallies into a short profile that
goes into every request's state. Clef cannot write text, so the profile is built from structured answers.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .clef import choice, noul, score

SURVEY_EVERY_S = 3600.0
HALF_LIFE_DAYS = 14.0
MIN_DAYS = 5  # no profile text for a slot until it has this many days of answers

PEOPLE = ["none", "one", "a few", "many"]
VEHICLES = {"none": "No vehicles.", "parked": "Only parked vehicles.", "moving": "At least one vehicle moving."}
SURVEY = {
    "_people": score("How many people are visible at the busiest moment of this clip?", PEOPLE),
    "_vehicles": choice("Which best describes vehicles in this clip?", VEHICLES),
    "_animal": noul("Is an animal visible at any point in this clip?"),
}


def slot_key(ts: float) -> str:
    t = datetime.fromtimestamp(ts)
    return f"{'weekend' if t.weekday() >= 5 else 'weekday'}-{t.hour:02d}"


def frequency(p: float) -> str:
    return "usually" if p >= 0.6 else "sometimes" if p >= 0.25 else "rarely"


@dataclass
class Slot:
    weights: dict[str, float] = field(default_factory=dict)  # "people:none" -> decayed probability mass
    total: float = 0.0  # decayed number of surveys
    days: list[str] = field(default_factory=list)  # distinct dates seen (most recent 60)
    updated: float = 0.0

    def decay(self, now: float) -> None:
        if self.updated:
            factor = 0.5 ** ((now - self.updated) / 86400 / HALF_LIFE_DAYS)
            self.weights = {k: v * factor for k, v in self.weights.items()}
            self.total *= factor
        self.updated = now

    def share(self, key: str) -> float:
        return self.weights.get(key, 0.0) / self.total if self.total else 0.0


class ProfileStore:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._slots: dict[str, dict[str, Slot]] = {}  # camera -> slot key -> Slot
        self._last_survey: dict[str, float] = {}
        if path and path.exists():
            raw = json.loads(path.read_text())
            self._slots = {cam: {k: Slot(**s) for k, s in slots.items()} for cam, slots in raw.get("slots", {}).items()}
            self._last_survey = raw.get("last_survey", {})

    def survey_due(self, camera: str, now: float) -> bool:
        with self._lock:
            last = self._last_survey.get(camera)
            if last is not None and now - last < SURVEY_EVERY_S:
                return False
            self._last_survey[camera] = now  # claim it, so parallel requests don't all survey
            return True

    def observe(self, camera: str, ts: float, answers: dict[str, dict[str, float]]) -> None:
        """Add one survey's answers (keys from SURVEY) for the window ending at ``ts``."""
        with self._lock:
            slot = self._slots.setdefault(camera, {}).setdefault(slot_key(ts), Slot())
            slot.decay(ts)
            slot.total += 1
            for level in PEOPLE:
                slot.weights[f"people:{level}"] = slot.weights.get(f"people:{level}", 0.0) + answers["_people"].get(level, 0.0)
            for option in VEHICLES:
                slot.weights[f"vehicles:{option}"] = slot.weights.get(f"vehicles:{option}", 0.0) + answers["_vehicles"].get(option, 0.0)
            slot.weights["animal"] = slot.weights.get("animal", 0.0) + answers["_animal"]["true"]
            day = datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
            if day not in slot.days:
                slot.days = (slot.days + [day])[-60:]
            self._save()

    def text(self, camera: str, ts: float) -> str:
        """Two or three sentences about what this camera usually sees in this hour slot, or "" if too little data."""
        with self._lock:
            slot = self._slots.get(camera, {}).get(slot_key(ts))
            if not slot or len(slot.days) < MIN_DAYS or not slot.total:
                return ""
            people = 1.0 - slot.share("people:none")
            crowd = slot.share("people:a few") + slot.share("people:many")
            moving = slot.share("vehicles:moving")
            animal = slot.share("animal")
        t = datetime.fromtimestamp(ts)
        when = f"{'weekends' if t.weekday() >= 5 else 'weekdays'} {t.hour:02d}:00-{(t.hour + 1) % 24:02d}:00"
        parts = [f"On {when}, people are {frequency(people)} present"
                 + (f" and groups {frequency(crowd)} appear." if people >= 0.25 else ".")]
        parts.append(f"Vehicles {frequency(moving)} move through.")
        if animal >= 0.25:
            parts.append(f"Animals are {frequency(animal)} seen.")
        return " ".join(parts)

    def unusual(self, camera: str, ts: float, person_probability: float) -> bool:
        """A person is likely present in a slot where people are rarely seen."""
        with self._lock:
            slot = self._slots.get(camera, {}).get(slot_key(ts))
            if not slot or len(slot.days) < MIN_DAYS:
                return False
            return person_probability >= 0.8 and (1.0 - slot.share("people:none")) < 0.25

    def reset(self, camera: str) -> None:
        with self._lock:
            self._slots.pop(camera, None)
            self._last_survey.pop(camera, None)
            self._save()

    def _save(self) -> None:
        if not self.path:
            return
        data = {"slots": {cam: {k: vars(s) for k, s in slots.items()} for cam, slots in self._slots.items()},
                "last_survey": self._last_survey}
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data))
        tmp.replace(self.path)


def person_probability(survey_answers: dict[str, dict[str, float]]) -> float:
    return 1.0 - survey_answers["_people"].get("none", 0.0)


