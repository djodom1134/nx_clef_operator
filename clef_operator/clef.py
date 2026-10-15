"""Client for the Clef server's SystemOne API (POST /v1/systemone, GET /health).

Question types: ``noul`` (true/false; the answer is P(true)), ``choice`` (criteria: {option: description}) and
``score`` (criteria: ordered list of levels). Every answer comes back as {option: probability}.
"""

from __future__ import annotations

import base64
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from . import env


@dataclass
class ClefResult:
    answers: dict[str, dict[str, float]]
    input_tokens: int
    seconds: float
    raw: dict = field(repr=False, default_factory=dict)


def noul(instructions: str) -> dict:
    return {"type": "noul", "instructions": instructions}


def choice(instructions: str, criteria: dict[str, str]) -> dict:
    return {"type": "choice", "instructions": instructions, "criteria": criteria}


def score(instructions: str, levels: list[str]) -> dict:
    return {"type": "score", "instructions": instructions, "criteria": levels}


def video_data_url(clip: bytes | str | Path) -> str:
    data = clip if isinstance(clip, bytes) else Path(clip).read_bytes()
    return "data:video/mp4;base64," + base64.b64encode(data).decode()


def image_data_url(image: bytes | str | Path, mime: str = "image/jpeg") -> str:
    data = image if isinstance(image, bytes) else Path(image).read_bytes()
    return f"data:{mime};base64," + base64.b64encode(data).decode()


def probabilities(answers: dict) -> dict[str, dict[str, float]]:
    out = {}
    for question_id, answer in answers.items():
        if answer["type"] == "noul":
            out[question_id] = {"true": answer["noul"], "false": 1 - answer["noul"]}
        else:
            out[question_id] = answer["probabilities"]
    return out


class Clef:
    def __init__(self, url: str | None = None, token: str | None = None, timeout: float = 600) -> None:
        self.url = (url or env.get("CLEF_URL", "http://192.168.87.134:8100")).rstrip("/")
        self.token = token if token is not None else env.get("CLEF_TOKEN")
        self.timeout = timeout

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def health(self) -> dict:
        with urllib.request.urlopen(self.url + "/health", timeout=10) as response:
            return json.load(response)

    def ask(
        self,
        state: str,
        questions: dict[str, dict],
        videos: list[bytes | str | Path] | None = None,
        video_fps: float | None = None,
        images: list[bytes | str | Path] | None = None,
    ) -> ClefResult:
        body: dict = {"model": "clef-omni", "state": state, "questions": questions}
        if images:
            body["images"] = [i if isinstance(i, str) and i.startswith("data:") else image_data_url(i) for i in images]
        if videos:
            body["videos"] = [v if isinstance(v, str) and v.startswith("data:") else video_data_url(v) for v in videos]
        if video_fps is not None:
            body["video_fps"] = video_fps
        request = urllib.request.Request(self.url + "/v1/systemone", data=json.dumps(body).encode(), headers=self._headers())
        started = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.load(response)
        except urllib.error.HTTPError as error:
            raise RuntimeError(f"Clef server {error.code}: {error.read().decode()[:500]}") from error
        return ClefResult(
            answers=probabilities(payload["answers"]),
            input_tokens=payload.get("usage", {}).get("input_tokens", 0),
            seconds=time.monotonic() - started,
            raw=payload,
        )
