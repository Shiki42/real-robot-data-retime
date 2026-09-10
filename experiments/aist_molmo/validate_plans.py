"""Pinned Molmo pilot. Experimental; no physical-success claim."""

import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from _common import run_stage, work_dir

from real_robot_data_retime.aist_timing import *


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
    result = []
    for ep in [15, 8]:
        d = root / f"ep{ep}"
        df = table[table.episode_index == ep].sort_values("frame_index")
        q = np.stack(df["observation.state"])
        a = np.stack(df["action"])
        for plan in json.loads((d / "screened_plans.json").read_text()):
            cfg = (
                TimingConfig()
                if plan["profile"] == "strict"
                else TimingConfig(
                    quantum_multiplier=4,
                    minimum_idle_seconds=0.25,
                    retained_idle_seconds=0.1,
                )
            )
            qt = estimate_tolerance(q, cfg)
            at = estimate_tolerance(a, cfg)
            protected = np.zeros(len(q), bool)
            protected[: plan["shared_prefix_last_frame"] + 1] = True
            protected[plan["shared_suffix_first_frame"] :] = True
            z = np.load(d / (plan["name"] + ".npz"))
            l, r = (z["left"], z["right"])
            for s, idx in [(0, l), (7, r)]:
                validate_map(
                    q[:, s : s + 7],
                    a[:, s : s + 7],
                    idx,
                    qt[s : s + 7],
                    at[s : s + 7],
                    protected,
                )
            ratios = {}
            for name, x in [("state", q), ("action", a)]:
                y = np.c_[x[l, :7], x[r, 7:]]
                ratios[name] = [
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
            if max(sum(ratios.values(), [])) > 1.00001:
                raise RuntimeError("New peak")
            result.append(
                dict(
                    episode=ep,
                    plan=plan["name"],
                    source_map_valid=True,
                    peak_ratios=ratios,
                )
            )
    (root / "screened_numeric_validation.json").write_text(json.dumps(result, indent=2))
    print("Validated", len(result), "screened plans", flush=True)


if __name__ == "__main__":
    run_stage(main, renderer=False, segmentation=False)
