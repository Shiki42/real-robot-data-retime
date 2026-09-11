# General smooth scheduling

Main video scheduling and joint-dataset scheduling apply smooth timing to
scheduled waits for every task profile (drawer, letters and workpiece).
New profiles using these planners receive the same behavior. The immutable
`modules/drawer_task` snapshot remains a historical reproduction, not a runtime
entry point. The independent uniform timing-grid augmenter is not a collision
planner and is not used as an alternate route here.

`timeline.smooth.speed_ramp` is shared by the accepted drawer lift-clock mode
and general waiting transitions. Default braking/restart durations are 0.5 s
and 0.3 s, rounded to output frame intervals. Integer source joins preserve
recorded frames outside ramps. Short segments between adjacent wait boundaries
reduce their peak rate so acceleration and braking do not overlap or reverse.
No-wait schedules retain their source clocks.

Drawer and letter general retiming uses a **coordinated paired-path clock**. At each arm's wait
entry or exit, both clocks ease to zero on the existing paired path, then
restart. Thus a partner arm may briefly decelerate even when it is not the arm
that needs to wait; it continues during the other arm's hold. This is a
conservative safety/complexity tradeoff, not a claim of minimum execution time.
The specialized accepted drawer high-point mode retains its existing independent
left-arm ramp and right-arm opening behavior. Its episode-0 source clocks remain
byte-identical to the accepted 536-frame preview.

Workpiece video scheduling instead uses independent approach clocks and fixed
staging poses. Both original approaches start together, including the full right
preparation; no separate preparation prefix delays the left arm. Waiting by one arm cannot retime an admitted execution by the
other arm. The first left execution and every admitted pickup/transport/place
interval retain source speed. See [workpiece priority](workpiece-alternating.md).

## Clearance and failure behavior

The post-retiming path must pass validation before export:

- The visual planner subdivides transitions at source-frame boundaries and
  checks swept projected foreground support and drawer precedence. Rendering
  also rejects new overlapping interpolated foregrounds without metric depth.
- The joint planner resamples both state and action, then runs mesh swept-edge
  checks on both. Drawer dependency gates, moving drawer-body clearance and
  the held-object proxy are checked on substeps with motion-bound slack.
- A contact-replay exception that was valid only for exact original paired
  edges cannot be stretched by the smoother; such a schedule is rejected.
- An unsafe smoothed schedule fails explicitly. It is never silently exported
  with hard stops. The current implementation does not search a second smooth
  trajectory after this rejection.

Projected checks remain projected checks, and interpolated depth is synthetic.
This does not certify physical calibration, hardware dynamics limits or
unmodeled obstacles. Source joint-path corners are retained; smoothing the
source clock does not remove all recorded acceleration discontinuities.

## Data and video correspondence

Both arm foregrounds, the moving drawer scene and wrist videos support
fractional source clocks through bidirectional optical flow. Main-view depth
uses the same RGB flow and valid-depth support, preserving uint16 millimetres.
Action/state use corresponding fractional row interpolation. Output source-frame
columns are float64; terminal holds preserve their fractional mapping dtype.
Auxiliary source records (capture timestamps, sensor/depth file references) are
explicitly retained from the floor source row; the exact synthetic time is in
`retime.left_source_frame` and `retime.right_source_frame`.

The dataset validator checks interpolated action/state after their stored dtype
conversion and reconstructs the expected fractional wrist frame. It no longer
silently truncates those clocks to integer source frames. Reports include
`smoothing.transitions`, durations, source path markers and output frame
boundaries. Discrete scheduling optimality does not describe the slower final
smoothed trajectory.

## Verification

On Coder A, 139 tests pass and 2 optional tests skip with the RoboVisualize
assets enabled. Coverage includes alternating waits, very short segments,
no-wait preservation, fractional actions/metadata, depth units, state/action
mesh-audit invocation, and fail-closed rejection of unsafe smoothed output.
Actual letters and workpiece checkpoint renders are under
`/home/coder/share/retime-general-smoothing-20260911`.

The letters regression rendered 1080 frames with 13 boundary transitions;
the workpiece regression rendered 882 frames with 7 transitions. Both passed
source-origin checks with zero interpolated foreground overlap. Transition
contact sheets were inspected. Machine-readable evidence is in
`verification.json` beside those renders.
