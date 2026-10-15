"""Small Nx Meta REST client: bearer-token login, cameras, motion masks, and live clip capture (needs ffmpeg)."""

from __future__ import annotations

import json
import ssl
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from . import env

# Nx servers use self-signed certificates by default.
_TLS = ssl.create_default_context()
_TLS.check_hostname = False
_TLS.verify_mode = ssl.CERT_NONE


@dataclass
class Camera:
    id: str
    name: str
    status: str
    vendor: str
    model: str


class Nx:
    def __init__(self, url: str | None = None, user: str | None = None, password: str | None = None) -> None:
        self.url = (url or env.get("NX_URL", "https://192.168.87.6:7001")).rstrip("/")
        self.user = user or env.get("NX_USER", "admin")
        self.password = password or env.get("NX_PASSWORD")
        self._token: str | None = None

    def _request(self, method: str, path: str, body: dict | None = None, auth: bool = True) -> dict | list:
        headers = {"Content-Type": "application/json"}
        if auth:
            headers["Authorization"] = f"Bearer {self.token}"
        data = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(self.url + path, data=data, headers=headers, method=method)
        with urllib.request.urlopen(request, timeout=30, context=_TLS) as response:
            raw = response.read()
        return json.loads(raw) if raw else {}

    @property
    def token(self) -> str:
        if self._token is None:
            if not self.password:
                raise RuntimeError("NX_PASSWORD is not set (environment or .env)")
            session = self._request(
                "POST", "/rest/v3/login/sessions",
                {"username": self.user, "password": self.password, "setCookie": False}, auth=False,
            )
            self._token = session["token"]
        return self._token

    def cameras(self, recording_only: bool = True) -> list[Camera]:
        devices = self._request("GET", "/rest/v3/devices?_with=id,name,status,vendor,model")
        cams = [Camera(d["id"].strip("{}"), d["name"], d["status"], d.get("vendor", ""), d.get("model", "")) for d in devices]
        return [c for c in cams if c.status == "Recording"] if recording_only else cams

    def device(self, camera_id: str) -> dict:
        return self._request("GET", f"/rest/v3/devices/{camera_id}")

    def motion_mask(self, camera_id: str) -> str:
        return self.device(camera_id).get("motion", {}).get("mask", "")

    def set_motion_mask(self, camera_id: str, mask: str) -> None:
        self._request("PATCH", f"/rest/v3/devices/{camera_id}", {"motion": {"mask": mask}})

    def engine_settings(self, engine_id: str) -> dict:
        return self._request("GET", f"/rest/v4/analytics/engines/{engine_id}/settings")

    def set_engine_settings(self, engine_id: str, updates: dict) -> dict:
        """Merge ``updates`` into an engine's (system page) settings."""
        return self._put_settings(f"/rest/v4/analytics/engines/{engine_id}/settings", updates)

    def device_agent_settings(self, engine_id: str, camera_id: str) -> dict:
        return self._request("GET", f"/rest/v4/analytics/engines/{engine_id}/deviceAgents/{camera_id}/settings")

    def set_device_agent_settings(self, engine_id: str, camera_id: str, updates: dict) -> dict:
        """Merge ``updates`` into a camera's settings for an engine (the camera page)."""
        return self._put_settings(f"/rest/v4/analytics/engines/{engine_id}/deviceAgents/{camera_id}/settings", updates)

    def _put_settings(self, path: str, updates: dict) -> dict:
        # The PUT must carry the model from the GET: without it the Server stores the values but hands the
        # running plugin its old ones, and a plugin reply with no model makes the Server drop every value.
        current = self._request("GET", path)
        values = {**current["values"], **updates}
        return self._request("PUT", path, {"values": values, "model": current["model"]})

    # PTZ: normalized pan/tilt in [-1, 1], zoom in [0, 1].
    def ptz_position(self, camera_id: str) -> dict:
        return self._request("GET", f"/rest/v4/devices/{camera_id}/ptz/position")

    def ptz_operational(self, camera_id: str) -> bool | None:
        """True if the Server can drive this camera's PTZ, False if it says it cannot, None if it could not tell
        (e.g. the Server is still starting)."""
        try:
            return self.ptz_position(camera_id).get("api") == "operational"
        except urllib.error.HTTPError as error:
            return False if error.code in (400, 404, 422) else None
        except Exception:  # noqa: BLE001 - not reachable right now
            return None

    def ptz_continuous(self, camera_id: str, pan: float, tilt: float, zoom: float) -> None:
        """Start (or with all zeros, stop) a continuous move; speeds in [-1, 1] are honoured, unlike absolute moves."""
        self._request("POST", f"/rest/v4/devices/{camera_id}/ptz/move", {"pan": pan, "tilt": tilt, "zoom": zoom})

    def ptz_absolute(self, camera_id: str, pan: float, tilt: float, zoom: float, speed: float = 0.1) -> None:
        self._request("POST", f"/rest/v4/devices/{camera_id}/ptz/position",
                      {"type": "absolute", "pan": pan, "tilt": tilt, "zoom": zoom, "speed": speed})

    def snapshot(self, camera_id: str) -> bytes:
        """The latest frame as a JPEG (full resolution)."""
        request = urllib.request.Request(f"{self.url}/rest/v4/devices/{camera_id}/image",
                                         headers={"Authorization": f"Bearer {self.token}"})
        with urllib.request.urlopen(request, timeout=30, context=_TLS) as response:
            return response.read()

    def generic_event(self, source: str, caption: str, description: str, camera_ids: list[str] | None = None) -> None:
        body = {"source": source, "caption": caption, "description": description}
        if camera_ids:
            body["deviceIds"] = camera_ids
        self._request("POST", "/rest/v4/events/generic", body)

    def capture_live(self, camera_id: str, seconds: float, out: Path, stream: int = 1) -> Path:
        """Record ``seconds`` of the live stream (0 = primary, 1 = secondary) to an MP4, without re-encoding."""
        out.parent.mkdir(parents=True, exist_ok=True)
        url = f"{self.url}/media/{camera_id}.mkv?stream={stream}"
        subprocess.run(
            ["ffmpeg", "-nostdin", "-loglevel", "error", "-tls_verify", "0",
             "-headers", f"Authorization: Bearer {self.token}\r\n",
             "-i", url, "-t", str(seconds), "-c", "copy", "-an", "-y", str(out)],
            check=True, timeout=seconds + 60,
        )
        return out


def probe(path: Path) -> dict:
    """Codec, size, duration and average fps of a clip (needs ffprobe)."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-count_packets", "-select_streams", "v:0",
         "-show_entries", "stream=codec_name,width,height,nb_read_packets:format=duration,size", "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    )
    info = json.loads(result.stdout)
    stream, fmt = info["streams"][0], info["format"]
    duration = float(fmt["duration"])
    frames = int(stream["nb_read_packets"])
    return {
        "codec": stream["codec_name"], "width": stream["width"], "height": stream["height"],
        "duration": duration, "frames": frames, "fps": frames / duration if duration else 0.0, "bytes": int(fmt["size"]),
    }
