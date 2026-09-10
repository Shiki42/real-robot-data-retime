"""Pinned Molmo pilot. Experimental; no physical-success claim."""

import json

import cv2
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from _common import run_stage, work_dir

from real_robot_data_retime.aist_timing import *
from real_robot_data_retime.decoupled_clock import compose_independent_sources


def main(args):
    """Run the pinned Molmo pilot stage; see README.md for input contracts."""
    P = work_dir()
    root = P / "experiments/molmo_video_v4"
    table = pd.concat(
        [
            pq.read_table(p).to_pandas()
            for p in (P / "data/molmo_block_pilot.download/data").rglob("*.parquet")
        ]
    )
    for ep in [15, 8]:
        d = root / f"ep{ep}"
        plans = json.loads((d / "plans.json").read_text())
        df = table[table.episode_index == ep].sort_values("frame_index")
        q = np.stack(df["observation.state"])
        a = np.stack(df["action"])
        n = len(q)
        packed = np.load(d / "robots.npz")["robots"]
        coarse = []
        for pair in packed:
            masks = np.unpackbits(pair, axis=-1, count=640)
            coarse.append(
                [
                    cv2.dilate(
                        (
                            cv2.resize(
                                m.astype(np.float32),
                                (80, 45),
                                interpolation=cv2.INTER_AREA,
                            )
                            > 0
                        ).astype(np.uint8),
                        np.ones((3, 3), np.uint8),
                    ).astype(bool)
                    for m in masks
                ]
            )
        coarse = np.array(coarse)
        coarse[:, :, 40:] = False
        outputs = []
        for profile, cfg in [
            ("strict", TimingConfig()),
            (
                "short_idle",
                TimingConfig(
                    quantum_multiplier=4,
                    minimum_idle_seconds=0.25,
                    retained_idle_seconds=0.1,
                ),
            ),
        ]:
            base = next((x for x in plans if x["name"] == profile + "_parallel"))
            p = base["shared_prefix_last_frame"]
            b = base["shared_suffix_first_frame"]
            protect = np.zeros(n, bool)
            protect[: p + 1] = True
            protect[b:] = True
            qt = estimate_tolerance(q, cfg)
            at = estimate_tolerance(a, cfg)
            seq = []
            for s in [0, 7]:
                idx, _ = compact_idle(
                    q[:, s : s + 7],
                    a[:, s : s + 7],
                    30,
                    qt[s : s + 7],
                    at[s : s + 7],
                    protect,
                    cfg,
                )
                seq.append(idx[(idx >= p) & (idx <= b)])
            samples = np.unique(
                np.r_[np.linspace(-len(seq[0]), len(seq[1]), 129).astype(int), 0]
            )
            rows = []
            best = None
            for delay in samples:
                l, r = compose_independent_sources(
                    *seq,
                    np.arange(p + 1),
                    np.arange(b, n),
                    left_delay=max(0, int(delay)),
                    right_delay=max(0, int(-delay)),
                )
                bad = (coarse[l, 0] & coarse[r, 1]).any((1, 2)) & (l != r)
                count = int(bad.sum())
                rows.append(dict(delay=int(delay), frames=len(l), overlap_frames=count))
                if not count and (best is None or len(l) < len(best[0])):
                    best = (l, r, int(delay))
            (d / (profile + "_phase_sweep.json")).write_text(json.dumps(rows, indent=2))
            if best is None:
                raise RuntimeError("Even sequential endpoint rejected")
            l, r, delay = best
            name = profile + "_parallel_screened"
            np.savez_compressed(d / (name + ".npz"), left=l, right=r)
            row = {
                **base,
                "name": name,
                "mode": "parallel_screened",
                "output_frames": len(l),
                "candidate_seconds": len(l) / 30,
                "speedup": n / len(l),
                "relative_start_delay_frames": delay,
                "phase_samples": len(samples),
                "projected_overlap_frames": 0,
                "projection_guard": "80x45 SAM3 union-partition masks, dilated 1 cell; shared base excluded",
                "scope": "Original uniform phase scheduler; source paths fixed; 2D projection screen only",
            }
            outputs.append(row)
            print("PHASE", ep, profile, len(l), delay, flush=True)
        (d / "screened_plans.json").write_text(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    run_stage(main, renderer=False, segmentation=False)
