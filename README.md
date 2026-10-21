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
| `clef_operator/gateway.py` | The gateway HTTP service (`/v1/jobs`, `/v1/classify`, `/v1/status`, `/health`, dashboard at `/`) |
| `clef_operator/dashboard.py` | The dashboard's data: answers per camera and question over time, Clef per minute |
| `clef_operator/static/dashboard.html` | The dashboard page (no external libraries: it runs inside Nx's browser) |
| `clef_operator/static/strip.html` | One camera's status strip (`/strip?camera=<id>`), to sit beside the camera in an Nx layout |
| `scripts/nx_strips.py` | Adds a Web Page per Clef camera's strip and the shared layout "Clef status" to Nx |
| `clef_operator/replay.py` | Offline replay: recording in, windows through the gateway, events out |
| `clef_operator/action.py` | Motion while a PTZ holds still: learns where a view always moves (trees), spots action to zoom in on |
| `clef_operator/embed.py` | EmbeddingGemma 2 client: a view's good-place share, tile cues (the hunter's cheap looks) |
| `clef_operator/embed_map.py` | Each PTZ's embedding map in (pan, tilt, field of view), searchable with plain text |
| `scripts/embed_server.py`, `scripts/embed.sh` | The EmbeddingGemma 2 server next to Clef (DGX, port 8300) |
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
- The hunter pans and tilts slowly (loud motors) and zooms quickly (silent): out before a pan, in after it. A zoom is
  one continuous command, timed, then a stop. Back PTZ zooms at one rate whatever the speed asked (about 10.5 degrees
  a second wide, 5 near 9 degrees; the time from a to b degrees is about 0.6 x |a^0.6 - b^0.6| s, fitted to zooms in
  from wide) and reads back a second or more late, so steering by the readback overshot by about 40% and each
  correction overshot back (13 <-> 25 degrees). It does not correct afterwards, and a zoom within 15% of the field of
  view already set is skipped. It used
  absolute zooms, but Back PTZ kept working on an absolute zoom-out after the pan that followed it cut it short, and
  zoomed itself in to 4-5 degrees with no further command (11:35 and 11:38 on 20 October: 37 to 3.7 degrees in 3 s,
  onto the lawn). That, not auto-tracking, was the close-up "fixation" on grass. After
  any zoom it sends nothing for `focus_s` (default 4 s, camera page "Focus time after zooming") while the camera
  refocuses. It decides its own zoom (a fixed share of time zoomed out, the earlier setting, kept it close up on
  schedule whether or not there was anything to look closer at). Wide is its stance: a wide view takes in the most.
  It looks around close up when it is bored of wide (each of its 5 most promising wide views watched in the last
  10 minutes, after at least 5 minutes wide; or 15 minutes wide): 3 close-up spots or 2 minutes, then wide again. It
  localizes on one tile when it needs to: Clef unsure about it, or the tile looking much more like what it looks for
  than the camera's views usually do (+0.08 similarity), not the same spot again within 10 minutes; then wide again.
  The dashboard says why it is zoomed in. Zooming in is for looking into the distance: from the camera's height
  (camera page, default 3.5 m) and how steeply it looks down at a spot, the hunter estimates how tall a person there
  would look, and only zooms where that is small, just far enough for the person to fill about a sixth of the
  frame (a spot 15 degrees down is about 13 m away from 3.5 m, where a person is already 7 degrees tall in the
  wide view: no zoom; 5 degrees down, about 40 m: zoom to about 19 degrees; near level: about 8). Before, its
  close-ups were mostly at 12-20 degrees down, of the grass by the picnic tables. Magnet views (home, presets and
  the operator's own) are always visited when their pull says so, whatever they look like: stay-away and hopeless
  scores never rule them out. First run
  (17:08): wide for 5 minutes, bored, 3 close-up spots in 2 minutes, wide again. Where a move arrived is judged
  once the zoom reads back as asked: a second after zooming in to 9 degrees Back PTZ still read 37, and the close-up
  look was learned as a look at the wide view it came from. Look-around close-ups are never tighter than 14 degrees
  and only where the distance calls for at least three quarters of that zoom; tighter ones are for evidence (Clef
  unsure, motion, the camera's detection). A view the camera reached zoomed in further than asked is learned at the
  field of view asked for, and learned views tighter than the minimum with no finds are dropped on load: each
  self-zoom had been learned as a new close-up and revisited, a ratchet that grew 72 views at 3-5 degrees.
- It holds a wide view it is worth watching (promising, not mostly unwanted) instead of hopping, and choosing the next
  view counts any view within half a field of view of a recent one as the same place: it used to ping-pong between
  neighbouring lawn views a few degrees apart.
- When the Server stops answering mid-hunt (a restart), the hunter stops the motor, waits for the PTZ and carries on.
- The hunter gets bored of places too:
  each recent visit (fading over 10 minutes) scales the appeal of nearby views by 0.4 (up to four visits' worth), so
  it moves between the areas likely to show the target. Negative prompts weigh on every choice (value = yield x
  (1 - avoid)), and views with no chance of a yes are never hunted: mostly wall, roof or sky (built-in blockers 0.5 or
  more), little open ground (scene under 0.15), or avoid answers at 0.7 or more, where the target was never found. On
  Back PTZ that rules out 22 of 35 wide views, the whole house side (pan 72-131) among them. Before, boredom was
  subtracted: once the good views were all bored, the side of the house (wall 0.68-0.78, under the old 0.8 veto)
  came out on top. Simulated on the learned views, the spot with finds (pan 175) gets about 1 look in 5 and the rest
  go to the open yard; nothing picked has an avoid score over 0.57. Views within a quarter of a field of view of
  each other are merged (513 learned views were near duplicates of 105).
- Operators and the camera's auto-tracker take precedence. The hunter checks every 2 s that the camera is where it
  left it (and, during a move, that axes it is not driving stay put); if someone else moved it, it stops without
  sending even a stop command and leaves the camera alone (no moves, snapshots or keep-alive) for `handback_s`
  (camera page "Take back after an operator or auto-tracking", default 10 s), restarting the count on every further
  move, then carries on from wherever the camera was left; the boredom cycle pauses meanwhile, and settings saved
  meanwhile keep the hand-over (and send no stop). Tested by panning Back PTZ through Nx: handed over within 2 s, the
  camera did not move until it had been still 10 s (180 s with the old default), then the hunter carried on. Each
  hand-over is logged. The Server's own idle return to its PTZ home (preset 300 on Back PTZ, which faces the house
  wall) also counts as someone else moving the camera; with a 10 s take-back the hunter drives it away again.
- A hand-over mid-move sends no stop (it would cut off the operator), so a false alarm left the hunter's own
  command running: Back PTZ panned in circles at its tilt end stop for 15 minutes, each move extending the
  hand-over. Now a takeover during a move counts only once the axes being driven stop following the hunter's command
  (a zoom readback still settling is not an operator), and after any hand-over the watcher checks the camera: still
  moving exactly as the hunter drove it (two checks, 2 s apart) means nobody replaced the command, so it stops the
  motor, logs it and carries on. Two false alarms are gone too: the zoom reads back seconds late, so after a zoom the watcher
  waits until the zoom reads as asked (it took the hunter's own zoom-out for an operator's); and the check before
  each snapshot compares with where the camera really stopped, not with the known view it was learned as (up to a
  quarter of a field of view away). Each hand-over is logged with what triggered it.
- Back PTZ slows pan and tilt as it zooms in (at 17 degrees of field of view, speed 0.2 pans about 2 degrees a
  second, not 6), so "the camera follows the hunter's command" means the commanded direction at 10-160% of the
  commanded rate; matching the rate closely took the hunter's own zoomed-in move for an operator's, then for nobody's
  (12:00, another runaway, stopped by hand). Last resort: a camera that keeps moving the same way for 60 s while
  handed over is stopped (a joystick or an auto-tracker does not drive one way for a minute).
- Clef tells well whether something is in any of several images, not which one: the PTZ hunter asks its 9 tiles in
  one request, and one by one only when that says maybe. An empty view costs 2 requests, not 10.
- While the PTZ hunter is moving a camera, the gateway drops that camera's windows: their motion is the camera's.
- Back PTZ reports its position about once a second, and moves pan 6, tilt 3.5 and zoom about 9.5 degrees a second at
  continuous speed 0.2. Its tilt end stop is +5 logical degrees (`tilt_max`); driven up into it, it pans by itself.
  The hunter stops a stalled axis, corrects in slower passes, and learns each view where the camera really ended up.
- Back PTZ sometimes ignores a continuous command that sets one axis to zero and keeps panning at full speed; only
  the full stop (0, 0, 0) is always obeyed. The hunter never changes one axis mid-move.
- Back PTZ has an Nx PTZ home position (preset 300): the Server drives it home after about 5 idle minutes, e.g.
  while hunter questions wait in a busy Clef queue. The hunter checks the camera is still where it left it before
  each snapshot, and does not learn from one taken elsewhere.
- For view scoring, "Is most of this view vegetation?" says yes to a lawn: the hunter asks for open ground a person
  could stand on, and treats walls, a roof or pergola below, and sky as blockers.
- The dashboard (`http://<gateway>:8200/`, in Nx as the Web Page "Clef dashboard" under Integrations; with a gateway
  token add `?token=...`) lists every camera in the Nx system (none are named in code) and shows the ones you pick:
  a live snapshot, every question's latest answer against its threshold, its probability over the last 1, 6 or 24
  hours with yes runs as lanes, a table view, the PTZ hunter's map of learned views, and Clef's load and time to
  answer per class against the targets. The selection and range live in the URL (`#cams=...&range=6`), so Nx layouts
  can pin different views. History (one point per answered window, one row per minute for Clef) is kept 24 h and
  saved in the state directory. Answers used to be drawn on the video as an analytics object per camera, but any
  object makes a camera that records on objects record, so they recorded all the time.
- The Nx Server Plugin SDK cannot draw text or graphs on video: a plugin only draws analytics objects (boxes, or
  points with `nx.sys.showAsPoint`, labelled with their attributes). So each camera's answers are a status strip,
  `/strip?camera=<id>`: a row per question (yes or no against its threshold, probability), its probability over the
  last hour with yes runs shaded, and the PTZ hunter's state. In Nx it is a Web Page per camera ("Clef · <camera>",
  under Integrations), and the shared layout "Clef status" puts each camera beside its strip
  (`scripts/nx_strips.py`; the local admin cannot create layouts owned by a cloud user, so the layout is shared).
  Clicking a graph in the Nx client plays that camera from then (the start of the answered window nearest the
  click): the page uses the Desktop Client JavaScript API (`window.vmsApiInit`, then `vms.tab.state()` and
  `vms.tab.setItemParams(item, {media: {timestampMs, speed}})`; `addItem` if the camera is not on the layout),
  which needs the Web Page's "Allow using Client API" (the pages are created with it).
- The hunter never jumps and sends no absolute moves: pan and tilt move only in slow continuous passes (to within 2
  degrees), zoom in quick continuous ones. While it holds still it sends a stop command every 60 s: the Server otherwise sends
  an idle PTZ to its home position at full speed after about 5 minutes (tested: with the keep-alive it stayed put).
- EmbeddingGemma 2 (`google/embeddinggemma-2`, text and image encoders, bf16, 70 soft tokens an image) runs next to
  Clef on the DGX (`scripts/embed.sh`, port 8300, 0.9 GB): one look (frame, 9 tiles cut by the server, a thumbnail)
  takes about 0.45 s while Clef keeps the GPU busy. Its Python is `~/embed/.venv`, which reuses Clef's CUDA torch
  through a `.pth` file and has its own transformers (5.20; the model needs it, Clef stays on 5.10). On the 37
  hand-labelled Back PTZ views (`bench/embed_hunter.py`) its good-place share (an open lawn, a path, a patio... against
  a wall, the roof, leaves close to the camera, the sky) ranks them with Spearman 0.75 (Clef's scene score 0.84); walls
  scored 0.01-0.25, open yard 0.38-0.82. It is not a person detector (labelled clip frames: AUC 0.90 overall, 0.53 on
  Front porch), so Clef still asks the hunt questions on every look. What it does for the hunter:
  - A view scoring under 0.10 is hopeless: Clef's 20 scene questions and its 9-tile request are skipped, and it is
    never hunted.
  - Zoom spots come from the look's tile vectors against the helper cues (no Clef request; it was 9, one per tile), and
    a tile that is itself hopeless does not become a zoomed view.
  - Every look goes into the camera's embedding map: the frame at its view and each tile at the direction it shows
    (centred on it, half the field of view), newest per spot, with a thumbnail (`~/.clef_operator/embed_map_<camera>`).
    The dashboard's hunter map has a search box: plain text ("picnic table", "the gate") colors every direction by
    similarity and lists the best matches, one per spot, with what matched and "Look there" (the hunter goes there
    next and watches). PTZ status strips have the same search, top three. API: `GET /v1/hunter_search?camera=&q=`,
    `GET /v1/hunter_thumb?camera=&shot=`, `POST /v1/hunter_look {camera_id, pan, tilt, fov}`.
  - If the embedding server is down, the hunter scores views and scans tiles with Clef as before (it retries after
    60 s). `--embed ''` turns it off.
- The hunter is set up with descriptions, not questions (camera page: "Look for", "Look closer at", "Stay away from",
  e.g. "person"; "path; gate"; "trees; wall"). Each look records how much the view looks like each (the best of the
  frame and its tiles for look-for and look-closer, the whole frame for stay-away), and a view's promise is how much
  more it looks like them than the camera's median view does: trees in every view of a yard do not rule them all
  out. Clef's scene questions are only the fallback when the embedding server is down; Clef still confirms every
  look, asked "Is there <description> in view?" (questions from older settings are asked as they are). The
  dashboard's hunter panel opens on where it is most worth looking: every direction scored by how much more it looks
  like what it looks for and where it looks closer than like what it stays away from (and walls, roofs, sky); with
  nobody in view, plain similarity to "person" ranked bushes first. Typing searches for anything else. "Found so far"
  is the older map of what Clef's answers showed at each learned view.
- The camera's own detections are feedback. The gateway polls Nx object tracks for each hunter camera every 2 s; a
  fresh track from the camera's analytics (Back PTZ: `nx.milesight.Person`, from the Milesight engine, whose
  `AutoTracking.Enable` is on) whose kind matches a "Look for" description ("person" also matches people, human,
  pedestrian...; "a van" matches Vehicle) is a find at the current view, and the camera's auto-tracking takes over:
  the hunter stops at once, even mid-move, and sends nothing while the camera keeps reporting the track, plus the
  take-back time. The camera's auto-tracking is switched off while the hunter hunts (it would chase things from
  views the hunter only passes through) and on while it follows a detection; the gateway finds the switch among the
  camera's analytics engine settings (`AutoTracking.Enable` on Back PTZ), keeps the original setting in
  `~/.clef_operator/autotracking.json` and restores it when the hunter stops or the gateway shuts down (or, after a
  crash, on the next start). Tracking stays on, and the hunter keeps its hands off, until 30 s after the last
  detection and after the camera last moved by itself: Back PTZ's person tracks last 1-8 s and break up while it
  follows someone, and switched off 10 s after the last one, its tracking was cut off mid-follow every time
  (15:41, 15:42 and 18:06 on 20 October; every one of the day's 8 person tracks did reach the hunter). The
  60 s runaway stop does not apply while it follows.
- Motion while the camera holds still. The plugin streams the Server's motion grid (44 x 32, OR-ed over half a
  second) to the gateway (`POST /v1/motion`) for hunter cameras, 2 a second. The hunter uses it only when the PTZ
  is still (not moving, refocusing, settling or handed over). Per view it learns each cell's share of frames with
  motion, kept across visits: cells moving in 12% of frames or more are boring (waving trees, IR noise at night).
  Action is a cluster of 2 or more other cells that keeps moving for a second and is new there (nothing near it 2-6 s
  before) or moves fast (1.5 cells a second); nothing counts until the view has been watched 30 s. Action cuts the
  hold short: the hunter goes there at full motor speed (no slow approach, no rest), zoomed for the motion's size and
  distance, and Clef looks. Where action was pulls the hunter's choice of view for half an hour, twice as much at
  night (19:00-07:00), when embeddings and Clef see less, but only action after which someone was found there
  pulls: chases of leaves found nobody, and pulling back to them kept the camera on infrared-lit trees after dark.
  A view first reached by a chase starts as promising as the known view nearest it, not as an unscored one (0.5,
  5-10 times the yard views around it). At most one chase every 30 s. Motion at or above the
  horizon is never chased (all of it above, or centred less than 2 degrees below): from a camera mounted above head
  height every person on the ground is below it, centred 2 degrees below at 75 m from 3.5 m up, so up there are
  only treetops, sky and rooflines (a tree line at tilt +2 drew a chase that zoomed in to 4.5 degrees, its foot at
  tilt 0 and -1.5 others to 6 and 14). The camera's own detections still find people further out.
  Leaves, bushes and shadows are learned by direction, not only per view (`action.NuisanceMap`, kept with the
  hunter's memory across restarts): every still frame teaches how often each 1-degree direction moves (6% of frames or more is
  a nuisance: gusty leaves moved in 9-11%), so leaves
  learned from one view are known from every view that overlaps it (each view learning its own took minutes at
  that view, and the hunter holds a view 20-45 s); and each chase that found nobody counts against its direction,
  fading over a day, so after two recent ones motion within about a metre of it on the ground (5 degrees out in
  the yard, 10-13 below the camera, where a bush spans far more) is not chased again until something
  is found there (gusts move leaves too seldom to count as busy, but each looked new: on Back PTZ the same far
  yard and bush spots drew a chase every few minutes, Clef 0.01 each time). The camera's own detections are
  never filtered.
- What not to look at. Each look's frame and its 9 tiles are classified (embed.ContentModel) as tree, bush, sky,
  siding (a house wall, window or door, a fence post close up), roof (or a pergola or lattice from above), grass or
  yard, from class descriptions plus the camera's own labelled images (7 nearest), on 256 of the 768 dimensions
  (as accurate, a third of the work). The operator's "Stay away from" descriptions are one more unwanted class. A
  view's unwanted coverage (frame and tiles averaged) scales down its promise, and at 0.75 or more it is never hunted.
  Close-ups are never taken of tiles or spots that are mostly unwanted or plain lawn: a tight zoom on grass shows
  nothing unless someone is there, and then motion, Clef or the camera's detection brings the hunter in. Measured on
  249 hand-labelled Back PTZ frames and tiles, day and infrared night (`bench/content_eval.py`, leave one out; labels
  in `bench/content_labels/`): 129 of 140 trees, bushes, sky, siding and pergola views caught, 10 of 109 open views
  wrongly flagged; plain lawn kept from close-ups 41 of 43, yard with something in it wrongly kept from them 15 of 66.
  The good-place share before it caught 69 (3 of 23 pergola or roof views, 4 of 11 bushes). Labelled examples live in
  `~/.clef_operator/content_labels_<camera>.json` (`bench/content_eval.py --save`); a camera without them uses the
  descriptions alone (124 caught, 22 wrongly flagged).
- Analytics events reach the Nx event log only when a rule uses them. Bookmark rules on these prolonged events
  need event state `none` and fire when the event ends.

## Dev environment

- Nx server: `Ubuntu_main`, `https://192.168.87.6:7001`, Nx Meta 6.1.2.42921, SSH alias `servy`. The Nx Server Plugin SDK 6.1.2 is unpacked at `~/nx_sdk/server_plugin_sdk` there, and its samples build.
- Clef server: DGX `http://192.168.87.134:8100`.
- Gateway: runs on the DGX next to Clef (`scripts/gateway.sh --clef http://127.0.0.1:8100`, port 8200, log
  `/tmp/clefop_gateway.log`, state in `~/.clef_operator`), started at boot by a `@reboot` crontab line after Clef has
  loaded. ffmpeg there is a static arm64 build in `~/bin`. The plugin's Gateway URL is `http://192.168.87.134:8200`.
- Embedding server: on the DGX too (`scripts/embed.sh`, port 8300, log `/tmp/embed_server.log`, model in
  `~/embed/hf`), started at boot by its own `@reboot` crontab line, a minute before the gateway.
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
