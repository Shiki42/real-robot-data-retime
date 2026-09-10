"""Pinned Molmo pilot. Experimental; no physical-success claim."""

import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from _common import run_stage, work_dir
from scipy.ndimage import median_filter

from real_robot_data_retime.aist_timing import *
from real_robot_data_retime.decoupled_clock import compose_independent_sources


def main(args):
    """Run the pinned Molmo pilot stage; see README.md for input contracts."""
    P = work_dir()
    root = P / "experiments/molmo_video_v4"
    root.mkdir(exist_ok=True)
    table = pd.concat(
        [
            pq.read_table(p).to_pandas()
            for p in (P / "data/molmo_block_pilot.download/data").rglob("*.parquet")
        ]
    )
    all_rows = []
    for ep in [15, 8]:
        df = table[table.episode_index == ep].sort_values("frame_index")
        q = np.stack(df["observation.state"])
        a = np.stack(df["action"])
        n = len(q)
        d = root / f"ep{ep}"
        d.mkdir(exist_ok=True)
        fps = 30
        smooth = median_filter(q[:, :6], size=(5, 1))
        rest = np.median(smooth[30:90], axis=0)
        away = np.max(abs(smooth - rest), axis=1) > 0.06
        runs = [(lo, hi) for lo, hi in ranges(away) if lo > fps * 3 and hi - lo >= 10]
        if not runs:
            raise RuntimeError("No left-stage onset")
        anchor = max(0, runs[0][0] - 12)
        rows = []
        for label, cfg, modes in [
            ("strict", TimingConfig(), ["left_first", "right_first", "parallel"]),
            (
                "short_idle",
                TimingConfig(
                    quantum_multiplier=4,
                    minimum_idle_seconds=0.25,
                    retained_idle_seconds=0.1,
                ),
                ["parallel"],
            ),
        ]:
            qt = estimate_tolerance(q, cfg)
            at = estimate_tolerance(a, cfg)
            quiet = np.array(
                [
                    np.all(np.ptp(q[max(0, t - 2) : min(n, t + 3)], axis=0) <= qt)
                    and np.all(np.ptp(a[max(0, t - 2) : min(n, t + 3)], axis=0) <= at)
                    for t in range(n)
                ]
            )
            initial = np.flatnonzero(quiet[:150])
            final = np.flatnonzero(quiet[max(0, n - 90) :]) + max(0, n - 90)
            p = int(initial[0]) if len(initial) else 0
            b = int(final[0]) if len(final) else n - 1
            protected = np.zeros(n, bool)
            protected[: p + 1] = True
            protected[b:] = True
            seq = []
            for side in [0, 7]:
                indices, _ = compact_idle(
                    q[:, side : side + 7],
                    a[:, side : side + 7],
                    fps,
                    qt[side : side + 7],
                    at[side : side + 7],
                    protected,
                    cfg,
                )
                seq.append(indices[(indices >= p) & (indices <= b)])
            for mode in modes:
                l, r = compose_independent_sources(
                    *seq, np.arange(p + 1), np.arange(b, n), mode
                )
                for side, idx in [(0, l), (7, r)]:
                    validate_map(
                        q[:, side : side + 7],
                        a[:, side : side + 7],
                        idx,
                        qt[side : side + 7],
                        at[side : side + 7],
                        protected,
                    )
                peaks = {}
                for name, x in [("state", q), ("action", a)]:
                    y = np.c_[x[l, :7], x[r, 7:]]
                    peaks[name] = [
                        float(
                            np.max(
                                np.max(abs(np.diff(y, n=k, axis=0)), axis=0)
                                / np.maximum(
                                    np.max(abs(np.diff(x, n=k, axis=0)), axis=0), 1e-09
                                )
                            )
                        )
                        for k in [1, 2]
                    ]
                name = label + "_" + mode
                np.savez_compressed(d / (name + ".npz"), left=l, right=r)
                row = dict(
                    episode=ep,
                    name=name,
                    mode=mode,
                    profile=label,
                    source_frames=n,
                    output_frames=len(l),
                    fps=fps,
                    source_seconds=n / fps,
                    candidate_seconds=len(l) / fps,
                    speedup=n / len(l),
                    shared_prefix_last_frame=p,
                    shared_suffix_first_frame=b,
                    hold_anchors_quiet=bool(quiet[p] and quiet[b]),
                    state_action_peak_ratios=peaks,
                    left_stage_reference_frame=anchor,
                    per_arm_core_frames=list(map(len, seq)),
                    scope="Unchanged per-arm source paths; composited visualization only; no object/collision certificate",
                    shared_exit_kind="Goal completion/parked suffix, not an assembly-contact event",
                )
                rows.append(row)
                all_rows.append(row)
        (d / "plans.json").write_text(json.dumps(rows, indent=2))
        print(
            "PLANS",
            ep,
            anchor,
            [(x["name"], x["output_frames"], x["hold_anchors_quiet"]) for x in rows],
            flush=True,
        )
    (root / "plans.json").write_text(json.dumps(all_rows, indent=2))


if __name__ == "__main__":
    run_stage(main, renderer=False, segmentation=False)
