# Staged drawer timing

The main video route accepts a matching per-episode joint parquet through
`--joint-data`, with explicit `--urdf` and `--mesh-root`. This enables the
pick/lift → wait for open → insert gate. Without metric inputs, video-only
scheduling retains its image-space gate; it cannot infer physical height.

The wait pose is selected between confirmed grasp and the image-space drawer
entry. Candidates must retain the grasp aperture and keep both arm meshes and
an assumed 35 mm held-object radius outside the inferred drawer sweep. Within
2 mm of the highest eligible TCP Z, the lowest 3-D source speed wins. Missing
safe held poses fail rather than substituting a pre-grasp wait.

Only a scheduled wait at that peak activates smoothing. The source clock slows
from unit speed to zero over 0.5 s and restarts from zero to unit speed over
0.3 s. Integrated monotone speed polynomials have zero clock acceleration at
both ramp endpoints. Integer source joins confine interpolation to the ramps.
At 30 FPS these are 15 and 9 intervals; other rates round to whole intervals.
The braking path must retain the grasp and clear the drawer sweep, including
quarter-source-frame samples of both state and action. Insertion starts only
after opening; closing starts only after withdrawal. An already-open drawer
allows uninterrupted insertion. Delays are measured from each arm's extracted
start pose, not from first handle contact or pickup.

`--left-delay-seconds` and `--right-delay-seconds` set nonnegative onset delays.
They allow parallel and ordered starts; the precedence gates still apply.
Only peak and terminal waits are allowed for left, and open/terminal waits for
right, so the scheduler cannot introduce a new abrupt mid-motion stop.

## Run

```bash
PYTHONPATH=src:/path/to/robo-visualize/src python main.py \
  --input episode.mp4 --output staged.mp4 --task drawer \
  --joint-data episode.parquet --urdf assets/piper_x_description.urdf \
  --mesh-root /path/to/robo-visualize/src/robo_visualize/arms/piperx/assets \
  --right-delay-seconds 2
```

The same joint and delay flags work with
`scripts/render_interaction_checkpoint.py --input ... --analysis ... --output ...`
to reuse a completed, source-verified analysis without rerunning segmentation.

The debug/output directory includes fractional `source_mapping.npz`,
`trajectories.parquet` (action, state, timestamp, both source clocks and an
interpolated flag), `trajectory_metrics.json`, and stage timing in `report.json`.
These are synchronized preview trajectories, not a newly assembled LeRobot
dataset. The existing automatic dataset exporter is a separate joint-collision
pipeline and does not invoke this video-stage mode.

RGB and telemetry use the same fractional source clocks. Joint rows use linear
interpolation along the original path; foreground RGB and masks use
bidirectional DIS optical flow. The moving drawer retains its integer right
clock. Fractional metric depth is not implemented and is rejected explicitly.

## Verification and scope

Coder A source: `Travor278/piperx-put-cube-in-drawer-20260908-87ep`, revision
`58bbbd720f6e78d162b8f4bc7077759d34c5162f`, episode 0, 634 frames at 30 FPS.
Reuse checkpoint: `retime-accuracy-20260910/drawer_0-final-photometric`.
Artifacts: `/home/coder/share/retime-staged-drawer-20260911/`.

The selected source frame is 322 (zero-based), TCP Z 0.233995 m. The delayed
preview exercises an airborne wait; the left-delayed preview exercises an
already-open drawer without inserting a stop. Both retain the original camera
compositor and its source-origin checks. Tests cover monotone timing, near-zero
stop/restart increments, integer joins, unsafe peak exclusion, interval-level
precedence, fixed gripper interpolation, and motion-compensated foregrounds.

The full moving two-arm audit remains projected silhouettes. Metric geometry
checks here cover the wait/braking pose against an inferred drawer proxy,
not calibrated full-scene collision freedom. Clock smoothing does not remove
recorded joint-path corners or certify hardware velocity/acceleration limits.
The generated video is inspected for transition artifacts, not certified as
physical robot execution. Existing drawer scene reconstruction limitations
remain recorded in the compositing report.

Final episode-0 checks: 129 tests passed, 4 skipped. `final-wait` has 536
frames, 22 fractional left frames, 0.5 s brake (output 142–157), a 62-interval
hold, and 0.3 s restart (219–228). `early` has 492 frames with no smoothing.
Both origin audits pass and both interval-level dependency checks pass.
Transition contact sheets were inspected; thin-cable/occlusion boundaries
remain subject to the source segmentation and optical-flow approximation.

## Local video/action alignment viewer

Build a self-contained comparison page from the two generated previews:

```bash
python scripts/build_action_preview.py /path/to/preview-root
python scripts/serve_action_preview.py --directory /path/to/preview-root --port 38765
```

The root contains `final-wait` and `early`, each with MP4, WebM, trajectories,
source mapping and report. The builder verifies every video's frame PTS against
trajectory timestamps (0.51 ms tolerance for WebM's millisecond time base),
and verifies both action source clocks against the renderer mapping. It embeds
all action rows and the exact video PTS in the page and writes a SHA-256 download
manifest. `serve.py` is copied alongside the page for local use.

The viewer displays J1–J6 for both arms plus gripper aperture on shared time axes.
`requestVideoFrameCallback` drives the displayed row from the presented frame's
media time; no independent animation clock or assumed wall-clock timing is used.
Seeking uses recorded PTS and updates curves only after a decoded frame is
presented. Curves, frame slider and previous/next controls seek the same video.
The range-capable loopback server is required for browser random access.

Local downloads for this run: `/Users/shuyuan/Downloads/drawer-action-alignment`.
Browser checks cover initial frame, frame 142 → 143, timeline end (535), chart
click seeking and playing/pausing. Both 536-frame and 492-frame media files pass
PTS checks. This exposes export alignment, not an independent proof that the
source robot state or optical-flow pose exactly matches its action command.
