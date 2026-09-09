"""Jointly select causal hypotheses with unique objects and consistent arm time."""

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp


def choose_episodes(candidates, object_count, task, fps):
    options = []
    for object_id in range(object_count):
        diverse = []
        for c in sorted(
            (x for x in candidates if x["object_id"] == object_id),
            key=lambda x: x["score"],
            reverse=True,
        ):
            if task == "drawer" and c["robot_id"] != "left":
                continue
            if any(
                c["robot_id"] == x["robot_id"]
                and abs(c["pickup_frame"] - x["pickup_frame"]) < fps * 0.5
                and abs(c["release_frame"] - x["release_frame"]) < fps * 0.5
                for x in diverse
            ):
                continue
            diverse.append(c)
            if len(diverse) == 6:
                break
        options.extend(diverse)
    if not options:
        return []
    count = len(options)
    rows = []
    limits = []
    for object_id in range(object_count):
        rows.append([float(c["object_id"] == object_id) for c in options])
        limits.append(1)
    for side in ["left", "right"]:
        rows.append([float(c["robot_id"] == side) for c in options])
        limits.append(1 if task == "drawer" else 2)
    for i, a in enumerate(options):
        for j in range(i + 1, count):
            b = options[j]
            if a["robot_id"] != b["robot_id"]:
                continue
            if (
                a["release_frame"] < b["grasp_start"]
                or b["release_frame"] < a["grasp_start"]
            ):
                continue
            row = np.zeros(count)
            row[i] = row[j] = 1
            rows.append(row)
            limits.append(1)
    # A unit increase in coverage dominates every possible score improvement.
    objective = -np.array([object_count + 1 + c["score"] for c in options])
    result = milp(
        objective,
        integrality=np.ones(count),
        bounds=Bounds(0, 1),
        constraints=LinearConstraint(np.asarray(rows), -np.inf, np.asarray(limits)),
        options={"time_limit": 30.0, "mip_rel_gap": 0.0},
    )
    if not result.success:
        raise RuntimeError(f"joint episode selection failed: {result.message}")
    chosen = [c for c, x in zip(options, result.x) if x > 0.5]
    return sorted(chosen, key=lambda c: c["pickup_frame"])
