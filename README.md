# Clef Operator

Plain-language Clef questions on Nx Meta camera video. The product spec is [docs/PRD.md](docs/PRD.md).

```
camera ──► Nx Server plugin (C++, plugin/)  ──jobs──►  Clef gateway (Python, clef_operator/)  ──►  Clef server (DGX :8100)
              30 s windows every 20 s                    priority queue, motion rank,
              motion grid, events, masks                 classifier, nuisance learning
```

## Layout

| Path | What |
| --- | --- |
| `clef_operator/clef.py` | Clef SystemOne client (`noul`, `choice`, `score` questions) |
| `clef_operator/nx.py` | Nx REST client: login, cameras, motion masks, live clip capture |
| `clef_operator/motion.py` | 44 × 32 motion grid: Nx mask parse/format, window motion summary, motion rank, clusters |
| `clef_operator/schedule.py` | Priority queue: class, then motion rank, then deadline; merging, expiry, promotion |
| `clef_operator/events.py` | Questions and the answer-to-event state machine |
| `clef_operator/classify.py` | Clef picks a question's fps, priority and whether it needs motion |
| `clef_operator/gateway.py` | The gateway HTTP service (`/v1/jobs`, `/v1/classify`, `/v1/status`, `/health`) |
| `clef_operator/replay.py` | Offline replay: recording in, windows through the gateway, events out |
| `clef_operator/hunter.py` | PTZ hunter: slow continuous-mode moves, survey, view scoring (positive and negative prompts), magnets, dwell |
| `clef_operator/profile.py` | Learned camera profiles (hourly survey tallies) |
| `bench/` | M0 benchmarks (timing per fps, resolution) |
| `tests/` | Unit tests (`uv run --group dev pytest`) |

The Python side has no third-party dependencies; it needs `ffmpeg` for clip work.

## Setup

```bash
cp .env.example .env          # fill in NX_PASSWORD; .env is git-ignored
uv run --python 3.12 --group dev pytest -q
```

## Running

```bash
# Gateway (in front of the Clef server)
python3 -m clef_operator.gateway --port 8200 --clef http://192.168.87.134:8100

# Replay a recording through it
python3 -m clef_operator.replay clip.mp4 --camera "Front porch" \
    --question "person:Is a person at the front door?:high:2"

# M0 timing benchmark (on a machine with ffmpeg that reaches Nx and Clef, e.g. the Nx server)
python3 -m bench.m0_timing --names "Front porch" "Driveway LPR" "Garage" --fps 1 2 4 6 8 10
```

## Plugin

```bash
# On the Nx server (SDK unpacked at ~/nx_sdk/server_plugin_sdk)
cmake -S plugin -B build -DnxSdkDir=$HOME/nx_sdk/server_plugin_sdk -DCMAKE_BUILD_TYPE=Release
cmake --build build -j && ./build/clef_operator_tests
sudo plugin/install.sh build/libclef_operator_plugin.so     # restarts the Nx server
journalctl -u networkoptix-metavms-mediaserver -f | grep "Clef Operator"
```

Settings can be changed in the Desktop Client or over REST with `Nx.set_engine_settings()` /
`Nx.set_device_agent_settings()`. Lessons from wiring it up on 6.1.2:

- A settings `PUT` must carry the `model` from the `GET`; without it the Server stores the values but the running
  plugin keeps its old ones.
- The plugin's settings reply must always include its model; a reply without one makes the Server drop every value.
- The Server's motion grid arrives attached to each video frame (`metadataList()`), not as separate packets.
- The Server names codecs MIME-style (`video/hevc`, `video/h264`); the gateway maps them to ffmpeg formats.
- Each event opens an object track (`nx.clef_operator.answer`) whose best shot is the frame with the most motion in
  the question's zone, boxed by the motion grid (zone or whole frame when nothing moved). The event carries the
  track id, so Nx shows that frame as its thumbnail, and answers are searchable in Object Search.
- Zone motion is counted per frame (frames with any motion inside the zone), not per cell: a car crossing a road
  zone in 2 s touches each cell for only a few frames and was being skipped as "no motion".
- The plugin submits a window's question groups in parallel, so the gateway's scheduler decides the order; sent one
  after another, a Normal question waited for the Critical answer before it even joined the queue, and expired.
- The gateway gathers a window's groups (0.3 s) and sends each zone's clip to Clef once, at the highest frame rate and
  priority asked for, with every question; each caller gets its own answers. Sent per group, the same 4 fps clip went
  to Clef once per priority (Front panorama's Critical and Normal, Garage Back's Critical and High): 30-50% of
  video requests were duplicates.
- Each job has a latency target from arrival (Critical 8 s, High 12 s, Normal 16 s, Low 20 s); the gateway runs the
  job due first and lowers frame rates (to 2, then 1 fps) when its simulated queue would miss one. `/v1/status` has
  `latency_1h` per class and `lowered_fps`. Replaying the busiest morning, every class stayed on target at up to 2x
  that load; before, High was answered within 20 s 86% of the time, and 22% at 1.5x.
- The PTZ hunter learns per view and hour how often looks there show the target, from its own snapshots and the
  camera's own windows while it holds the view; views that stay empty go quiet (skipped, rechecked every 3 hours).
  `/v1/hunter` lists the best and quiet views.
- Hunter helper questions (camera page, "Look closer at places with"; default path, gate, play area, seating area,
  yard) mark places where the hunt questions are more often yes. A view's score rises only where a helper answer
  stands out from the camera's usual answers ("yard" is yes almost everywhere and counts for nothing). Every 6 hours
  the hunter asks each tile of a promising wide view the helpers; a tile that stands out (at least 0.5, and 0.25
  above the view's median tile) becomes a zoomed view at half the field of view, learned like any other. On Back PTZ
  at pan 190 the picnic-table tiles stood out at 0.90-0.95 and the path tiles at 0.72-0.75.
- The hunter pans and tilts slowly (loud motors) but zooms in one step (silent): out before a pan, in after it. After
  any zoom it sends nothing for `focus_s` (default 4 s, camera page "Focus time after zooming") while the camera
  refocuses. It spends `wide_share` of its time zoomed all the way out (default 75%, "Time zoomed all the way out");
  the rest it looks closer, at zoomed views found by the helper questions, two levels deep (about 18 and 9 degrees).
- When the Server stops answering mid-hunt (a restart), the hunter stops the motor, waits for the PTZ and carries on.
- Clef tells well whether something is in any of several images, not which one: the PTZ hunter asks its 9 tiles in
  one request, and one by one only when that says maybe. An empty view costs 2 requests, not 10.
- While the PTZ hunter is moving a camera, the gateway drops that camera's windows: their motion is the camera's.
- Back PTZ reports its position about once a second, and moves pan 6, tilt 3.5 and zoom about 9.5 degrees a second at
  continuous speed 0.2. Its tilt end stop is +5 logical degrees (`tilt_max`); driven up into it, it pans by itself.
  The hunter stops a stalled axis, corrects in slower passes, and learns each view where the camera really ended up.
- Back PTZ sometimes ignores a continuous command that sets one axis to zero and keeps panning at full speed; only
  the full stop (0, 0, 0) is always obeyed. The hunter never changes one axis mid-move, and finishes the last few
  degrees with an absolute move (exact, but it ignores its speed).
- Back PTZ has an Nx PTZ home position (preset 300): the Server drives it home after about 5 idle minutes, e.g.
  while hunter questions wait in a busy Clef queue. The hunter checks the camera is still where it left it before
  each snapshot, and does not learn from one taken elsewhere.
- For view scoring, "Is most of this view vegetation?" says yes to a lawn: the hunter asks for open ground a person
  could stand on, and treats walls, a roof or pergola below, and sky as blockers.
- Each camera with questions shows the latest answer to every question, small in the lower right of the video: one
  `nx.clef_operator.answers` object (type flag `nonIndexable`: live and playback, never in Object Search) with a
  tiny box in the corner and one attribute per question ("YES 0.92", "no 0.02", "closed 0.85", "no motion", or its
  age when older than a minute), refreshed every second. System setting "Show answers on video" turns it off.
- Analytics events reach the Nx event log only when a rule uses them. Bookmark rules on these prolonged events
  need event state `none` and fire when the event ends.

## Dev environment

- Nx server: `Ubuntu_main`, `https://192.168.87.6:7001`, Nx Meta 6.1.2.42921, SSH alias `servy`. The Nx Server Plugin SDK 6.1.2 is unpacked at `~/nx_sdk/server_plugin_sdk` there, and its samples build.
- Clef server: DGX `http://192.168.87.134:8100`.
- Gateway: runs on the DGX next to Clef (`scripts/gateway.sh --clef http://127.0.0.1:8100`, port 8200, log
  `/tmp/clefop_gateway.log`, state in `~/.clef_operator`), started at boot by a `@reboot` crontab line after Clef has
  loaded. ffmpeg there is a static arm64 build in `~/bin`. The plugin's Gateway URL is `http://192.168.87.134:8200`.
- DGX Spark (Nx in docker `nx-dgx-spark`, host networking, 192.168.87.134:7001) runs the arm64 plugin, built natively in
  `~/nx_clef_operator/build`. It is installed by `/srv/mediaserver/entrypoint.d/20-clef-operator.sh`
  (`plugin/docker-entrypoint-20-clef-operator.sh`), which copies `entrypoint.d/clef_operator/libclef_operator_plugin.so`
  into the plugins folder on every container start, since that folder is not a volume.
- Clef runs its MoE experts (91% of the weights) in FP8: `CLEF_FP8=1` in `~/Code/clef-trader/.clef-server.env`, read by
  `deploy/clef_server_supervise.sh` when it starts, so restart the supervisor (not only the server) after changing it.
  Code: `clef_trader/fp8_experts.py` (row-scaled FP8 GEMMs with `torch._scaled_mm`; GB10 has no grouped FP8 GEMM).
  Measured with `bench/clef_precision.py` on the same 105 inputs: GPU memory 77.5 -> 60.9 GB, 4 fps clip 2.66 -> 2.46 s,
  snapshot 0.59 -> 0.51 s; person accuracy 100% both, PTZ view ranking 0.848 -> 0.843, median answer change 0.001.
  Roll back: delete the line, kill the supervisor and the server, start the supervisor again.
- Stopping the gateway (SIGTERM) stops every PTZ motor first; hunters resume from `hunters.json` on the next start.
- Cameras with Clef: Front porch, Front panorama and Back PTZ (on DGX Spark) and Garage Back (on Ubuntu_main). Rules
  turn Clef events into desktop notifications and bookmarks (tag `clef`). Analytics cameras record objects only.
