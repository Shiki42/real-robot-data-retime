"""Joint input and synchronized trajectory artifacts for the main video route."""

import json
from pathlib import Path
import numpy as np
from .timeline.smooth import sample_rows


def load_joints(path, urdf, mesh_root, timeline):
    import pyarrow.parquet as pq

    if urdf is None or mesh_root is None:
        raise ValueError("joint staging requires URDF and mesh root")
    table = pq.read_table(path)
    if len(table) != timeline["source_frames"]:
        raise ValueError("joint table and video frame counts differ")
    if len(set(table["episode_index"].to_pylist())) != 1:
        raise ValueError("joint table must contain exactly one episode")
    indices = np.asarray(table["frame_index"].to_pylist())
    if not np.all(np.diff(indices) == 1):
        raise ValueError("joint table frames must be contiguous and ordered")
    times = np.asarray(table["timestamp"].to_pylist(), float)
    if not np.allclose(np.diff(times), 1 / timeline["fps"], atol=1e-4):
        raise ValueError("joint and video sampling rates differ")
    return (
        np.asarray(table["observation.state"].to_pylist(), float),
        np.asarray(table["action"].to_pylist(), float),
        urdf,
        mesh_root,
    )


def export_trajectories(output, joints, left, right, fps, plan):
    import pyarrow as pa
    import pyarrow.parquet as pq

    output = Path(output)
    mapped = {}
    for name, values in zip(("observation.state", "action"), joints[:2]):
        mapped[name] = np.concatenate(
            (sample_rows(values[:, :7], left), sample_rows(values[:, 7:], right)),
            axis=1,
        )
    table = {
        name: pa.array(values.tolist(), type=pa.list_(pa.float64(), 14))
        for name, values in mapped.items()
    }
    table.update(
        timestamp=pa.array(np.arange(len(left)) / fps),
        left_source_frame=pa.array(np.asarray(left, float)),
        right_source_frame=pa.array(np.asarray(right, float)),
        interpolated=pa.array((left != np.floor(left)) | (right != np.floor(right))),
    )
    pq.write_table(pa.table(table), output / "trajectories.parquet")
    metrics = {}
    for name, values in mapped.items():
        velocity = np.diff(values[:, :6], axis=0) * fps
        acceleration = np.diff(velocity, axis=0) * fps
        metrics[name] = dict(
            left_max_joint_speed_deg_s=np.abs(velocity).max(axis=0).tolist(),
            left_max_joint_acceleration_deg_s2=np.abs(acceleration)
            .max(axis=0)
            .tolist(),
        )
    stages = plan["stages"]
    if stages["smooth_stop_required"]:
        start = stages["brake_start_output_frame"]
        stop = stages["stop_output_frames"][0]
        resume = stages["restart_start_output_frame"]
        end = stages["restart_end_output_frame"]
        for name, values in mapped.items():
            speed = np.linalg.norm(np.diff(values[:, :6], axis=0) * fps, axis=1)
            metrics[name]["transition_joint_speed_norm_deg_s"] = dict(
                brake_first=float(speed[start]),
                brake_last=float(speed[stop - 1]),
                restart_first=float(speed[resume]),
                restart_last=float(speed[end - 1]),
                hold_max=float(speed[stop:resume].max()) if resume > stop else 0,
            )
    (output / "trajectory_metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics
