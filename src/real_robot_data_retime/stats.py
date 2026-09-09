from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

NUMERIC_FEATURES = ("action", "observation.state")
QUANTILES = {
    "q01": 0.01,
    "q10": 0.10,
    "q50": 0.50,
    "q90": 0.90,
    "q99": 0.99,
}


def feature_statistics(values: np.ndarray) -> dict[str, list[float] | list[int]]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or len(array) == 0:
        raise ValueError(f"statistics expect a non-empty 2D array, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("statistics input contains non-finite values")
    stats: dict[str, list[float] | list[int]] = {
        "min": array.min(axis=0).tolist(),
        "max": array.max(axis=0).tolist(),
        "mean": array.mean(axis=0).tolist(),
        "std": array.std(axis=0).tolist(),
        "count": [len(array)],
    }
    stats.update(
        {
            name: np.quantile(array, probability, axis=0).tolist()
            for name, probability in QUANTILES.items()
        }
    )
    return stats


def compute_dataset_numeric_stats(
    dataset: Path,
    features: Iterable[str] = NUMERIC_FEATURES,
) -> tuple[dict[str, dict[str, list[float] | list[int]]], list[Path]]:
    root = dataset.resolve()
    selected = tuple(features)
    if not selected:
        raise ValueError("at least one numeric feature is required")
    data_files = sorted((root / "data").rglob("*.parquet"))
    if not data_files:
        raise FileNotFoundError(f"no data parquet files found under {root / 'data'}")

    chunks: dict[str, list[np.ndarray]] = {feature: [] for feature in selected}
    widths: dict[str, int] = {}
    total_rows = 0
    for path in data_files:
        table = pq.read_table(path, columns=list(selected))
        total_rows += table.num_rows
        for feature in selected:
            values = _numeric_column_array(table[feature], feature, path)
            width = values.shape[1]
            previous = widths.setdefault(feature, width)
            if width != previous:
                raise ValueError(
                    f"feature {feature} width changed from {previous} to {width} in {path}"
                )
            chunks[feature].append(values)
    if total_rows == 0:
        raise ValueError(f"dataset has no numeric rows: {root}")

    info_path = root / "meta" / "info.json"
    if info_path.is_file():
        expected_rows = int(json.loads(info_path.read_text())["total_frames"])
        if expected_rows != total_rows:
            raise ValueError(
                f"data row count {total_rows} does not match info total_frames {expected_rows}"
            )
    stats = {
        feature: feature_statistics(np.concatenate(chunks[feature], axis=0))
        for feature in selected
    }
    return stats, data_files


def rewrite_dataset_numeric_stats(
    dataset: Path,
    features: Iterable[str] = NUMERIC_FEATURES,
) -> dict[str, Any]:
    root = dataset.resolve()
    stats_path = root / "meta" / "stats.json"
    if not stats_path.is_file():
        raise FileNotFoundError(f"LeRobot stats metadata missing: {stats_path}")
    numeric_stats, data_files = compute_dataset_numeric_stats(root, features)
    stats = json.loads(stats_path.read_text())
    stats.update(numeric_stats)
    _atomic_write_json(stats_path, stats)
    frame_counts = {
        feature: int(value["count"][0]) for feature, value in numeric_stats.items()
    }
    if len(set(frame_counts.values())) != 1:
        raise ValueError(f"numeric feature frame counts differ: {frame_counts}")
    return {
        "dataset": str(root),
        "data_files": [str(path.relative_to(root)) for path in data_files],
        "frame_count": next(iter(frame_counts.values())),
        "features": list(numeric_stats),
    }


def create_corrected_stats_copy(
    source: Path,
    output: Path,
    *,
    copy_mode: str = "hardlink",
) -> dict[str, Any]:
    source_root = source.resolve()
    output_root = output.resolve()
    if not (source_root / "meta" / "stats.json").is_file():
        raise FileNotFoundError(f"LeRobot dataset metadata missing: {source_root}")
    if output_root.exists():
        raise FileExistsError(f"output already exists: {output_root}")
    if copy_mode not in {"hardlink", "copy"}:
        raise ValueError(f"unsupported copy mode: {copy_mode}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = output_root.parent / f".{output_root.name}.stats-staging-{uuid4().hex}"
    copy_function = os.link if copy_mode == "hardlink" else shutil.copy2
    try:
        shutil.copytree(source_root, staging, copy_function=copy_function)
        receipt = rewrite_dataset_numeric_stats(staging)
        os.replace(staging, output_root)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    receipt.update(
        {
            "source_dataset": str(source_root),
            "dataset": str(output_root),
            "copy_mode": copy_mode,
        }
    )
    return receipt


def _numeric_column_array(
    column: pa.ChunkedArray,
    feature: str,
    path: Path,
) -> np.ndarray:
    combined = column.combine_chunks()
    if combined.null_count:
        raise ValueError(f"feature {feature} contains null values in {path}")
    if pa.types.is_fixed_size_list(combined.type):
        width = combined.type.list_size
        values = combined.values.to_numpy(zero_copy_only=False)
        array = np.asarray(values, dtype=np.float64).reshape(len(combined), width)
    else:
        array = np.asarray(combined.to_pylist(), dtype=np.float64)
        if array.ndim == 1:
            array = array[:, None]
    if array.ndim != 2:
        raise ValueError(f"feature {feature} in {path} is not fixed-width numeric data")
    return array


def _atomic_write_json(path: Path, value: Any) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.staging-",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(path.stat().st_mode)
        os.replace(temporary, path)
    except Exception:
        if temporary is not None and temporary.exists():
            temporary.unlink()
        raise
