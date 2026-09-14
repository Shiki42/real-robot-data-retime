# Conditional lift stopping

Uniform drawer planning previously inserted the 0.5s braking and0.3s restart
ramps unconditionally. It now first computes the relative timing using the
recorded-speed lift. When the drawer is already open at natural peak arrival,
the original left source clock advances by one frame across the lift peak.
Only late-opening schedules use the smoothed stop and recompute their actual
lift duration. Uniform positions and pair spacing remain unchanged; lift
prerequisite duration may differ across variants because only one needs ramps.
Right closing waits and their smoothing remain active in either mode.

All78 sources /156 variants replanned successfully.61 variants now pass the peak
without stopping.209 tests passed,4 skipped. Original source26/27/29/30 were
rendered in both variants and passed numeric Action/State and all-camera frame
validation. This includes reported outputs100–103, whose source clocks advance
exactly1 frame per output frame around the peak with the drawer already open.

Run root: /home/coder/share/drawer-no-unneeded-stop-20260915
The156 old review videos at localhost:38770 are unchanged. Only the8 pilot
outputs are rendered in the new root; this is not a fully regenerated dataset.
Any full regeneration must export masks again against its new output clocks.
