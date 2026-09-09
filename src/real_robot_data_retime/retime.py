from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

ARM_SLICES = {"left": slice(0, 7), "right": slice(7, 14)}


@dataclass(frozen=True)
class MotionHeuristic:
    smoothing_frames: int = 7
    noise_floor_per_frame: float = 0.08
    search_start_fraction: float = 0.2
    search_end_fraction: float = 0.8
    boundary_energy_fraction: float = 0.005
    boundary_padding_frames: int = 3
    minimum_segment_frames: int = 30
    maximum_cross_motion_ratio: float = 0.1


@dataclass(frozen=True)
class ArmSegment:
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class EpisodeSegments:
    split: int
    left: ArmSegment
    right: ArmSegment
    state_right_before_split_ratio: float
    state_left_after_split_ratio: float
    action_right_before_split_ratio: float
    action_left_after_split_ratio: float

    def receipt(self) -> dict[str, object]:
        return {
            "split": self.split,
            "left": asdict(self.left),
            "right": asdict(self.right),
            "static_gap_frames": self.right.start - self.left.end,
            "state_right_before_split_ratio": self.state_right_before_split_ratio,
            "state_left_after_split_ratio": self.state_left_after_split_ratio,
            "action_right_before_split_ratio": self.action_right_before_split_ratio,
            "action_left_after_split_ratio": self.action_left_after_split_ratio,
        }


@dataclass(frozen=True)
class FrameRange:
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start

    def seconds(self, fps: float) -> dict[str, float]:
        return {"start": self.start / fps, "end": self.end / fps}


@dataclass(frozen=True)
class RetimePlan:
    left_source_indices: np.ndarray
    right_source_indices: np.ndarray
    left_active: np.ndarray
    right_active: np.ndarray
    left_start_frame: int
    right_start_frame: int
    grid_index: int
    grid_size: int
    schedule_position_frame: int

    @property
    def length(self) -> int:
        return len(self.left_source_indices)

    def timing_receipt(self, fps: float) -> dict[str, object]:
        overlap = boolean_ranges(self.left_active & self.right_active)
        left_idle = boolean_ranges(~self.left_active)
        right_idle = boolean_ranges(~self.right_active)
        left_execution = FrameRange(
            self.left_start_frame,
            self.left_start_frame + int(np.count_nonzero(self.left_active)),
        )
        right_execution = FrameRange(
            self.right_start_frame,
            self.right_start_frame + int(np.count_nonzero(self.right_active)),
        )
        return {
            "grid_index": self.grid_index,
            "grid_size": self.grid_size,
            "normalized_position": self.grid_index / self.grid_size,
            "schedule_position_frame": self.schedule_position_frame,
            "left_start_frame": self.left_start_frame,
            "right_start_frame": self.right_start_frame,
            "left_start_seconds": self.left_start_frame / fps,
            "right_start_seconds": self.right_start_frame / fps,
            "left_execution_frames": asdict(left_execution),
            "right_execution_frames": asdict(right_execution),
            "left_execution_seconds": left_execution.seconds(fps),
            "right_execution_seconds": right_execution.seconds(fps),
            "left_idle_frames": [asdict(item) for item in left_idle],
            "right_idle_frames": [asdict(item) for item in right_idle],
            "overlap_frames": [asdict(item) for item in overlap],
            "left_idle_seconds": [item.seconds(fps) for item in left_idle],
            "right_idle_seconds": [item.seconds(fps) for item in right_idle],
            "overlap_seconds": [item.seconds(fps) for item in overlap],
            "both_idle_frames": int(
                np.count_nonzero(~self.left_active & ~self.right_active)
            ),
        }


def arm_motion_energy(
    values: np.ndarray,
    arm: str,
    heuristic: MotionHeuristic,
) -> np.ndarray:
    vector = _validate_vector(values)
    if arm not in ARM_SLICES:
        raise ValueError(f"unknown arm: {arm}")
    per_frame_step = np.max(np.abs(np.diff(vector[:, ARM_SLICES[arm]], axis=0)), axis=1)
    kernel = np.ones(heuristic.smoothing_frames) / heuristic.smoothing_frames
    smoothed = np.convolve(per_frame_step, kernel, mode="same")
    return np.maximum(smoothed - heuristic.noise_floor_per_frame, 0.0)


def detect_arm_segments(
    state: np.ndarray,
    action: np.ndarray,
    heuristic: MotionHeuristic = MotionHeuristic(),
) -> EpisodeSegments:
    state_vector = _validate_vector(state)
    action_vector = _validate_vector(action)
    if state_vector.shape != action_vector.shape:
        raise ValueError("state and action must have identical shapes")
    _validate_heuristic(heuristic)

    state_left = arm_motion_energy(state_vector, "left", heuristic)
    state_right = arm_motion_energy(state_vector, "right", heuristic)
    split = _minimum_cross_energy_split(state_left, state_right, heuristic)
    left = _energy_segment(state_left, 0, split, heuristic)
    right = _energy_segment(state_right, split, len(state_right), heuristic)
    left, right = _keep_segments_disjoint(left, right)

    state_ratios = _cross_motion_ratios(state_left, state_right, split)
    action_left = arm_motion_energy(action_vector, "left", heuristic)
    action_right = arm_motion_energy(action_vector, "right", heuristic)
    action_ratios = _cross_motion_ratios(action_left, action_right, split)
    ratios = (*state_ratios, *action_ratios)
    if max(ratios) > heuristic.maximum_cross_motion_ratio:
        raise ValueError(
            "episode is not sufficiently left-then-right decoupled: "
            f"cross-motion ratios={ratios}"
        )
    if min(left.length, right.length) < heuristic.minimum_segment_frames:
        raise ValueError(
            f"detected arm segment is too short: left={left.length} right={right.length}"
        )
    return EpisodeSegments(
        split=split,
        left=left,
        right=right,
        state_right_before_split_ratio=state_ratios[0],
        state_left_after_split_ratio=state_ratios[1],
        action_right_before_split_ratio=action_ratios[0],
        action_left_after_split_ratio=action_ratios[1],
    )


def build_uniform_schedule_plan(
    segments: EpisodeSegments,
    grid_index: int,
    grid_size: int,
) -> RetimePlan:
    if grid_size < 1 or not 0 <= grid_index <= grid_size:
        raise ValueError(f"grid index must be in [0, {grid_size}], got {grid_index}")
    total_duration = segments.left.length + segments.right.length
    schedule_position = _round_fraction(grid_index * total_duration, grid_size)
    if schedule_position <= segments.left.length:
        left_start = 0
        right_start = segments.left.length - schedule_position
    else:
        left_start = schedule_position - segments.left.length
        right_start = 0

    output_length = max(
        left_start + segments.left.length,
        right_start + segments.right.length,
    )
    timeline = np.arange(output_length, dtype=np.int64)
    left_active = _active_mask(timeline, left_start, segments.left.length)
    right_active = _active_mask(timeline, right_start, segments.right.length)
    if np.any(~left_active & ~right_active):
        raise ValueError("retime schedule introduced a both-idle gap")
    return RetimePlan(
        left_source_indices=_scheduled_indices(timeline, left_start, segments.left),
        right_source_indices=_scheduled_indices(timeline, right_start, segments.right),
        left_active=left_active,
        right_active=right_active,
        left_start_frame=left_start,
        right_start_frame=right_start,
        grid_index=grid_index,
        grid_size=grid_size,
        schedule_position_frame=schedule_position,
    )


def boolean_ranges(mask: np.ndarray) -> list[FrameRange]:
    values = np.asarray(mask, dtype=bool)
    if values.ndim != 1:
        raise ValueError("range mask must be one-dimensional")
    changes = np.diff(np.concatenate(([False], values, [False])).astype(np.int8))
    return [
        FrameRange(int(start), int(end))
        for start, end in zip(
            np.flatnonzero(changes == 1),
            np.flatnonzero(changes == -1),
        )
    ]


def materialize_dual_arm(values: np.ndarray, plan: RetimePlan) -> np.ndarray:
    vector = _validate_vector(values)
    if max(plan.left_source_indices.max(), plan.right_source_indices.max()) >= len(
        vector
    ):
        raise IndexError("retime source index is outside the episode")
    return np.concatenate(
        (
            vector[plan.left_source_indices, ARM_SLICES["left"]],
            vector[plan.right_source_indices, ARM_SLICES["right"]],
        ),
        axis=1,
    )


def _minimum_cross_energy_split(
    left: np.ndarray,
    right: np.ndarray,
    heuristic: MotionHeuristic,
) -> int:
    left_total = _positive_total(left, "left")
    right_total = _positive_total(right, "right")
    start = max(1, round(len(left) * heuristic.search_start_fraction))
    end = min(len(left) - 1, round(len(left) * heuristic.search_end_fraction))
    if start >= end:
        raise ValueError("episode is too short for the configured split search window")
    candidates = np.arange(start, end, dtype=np.int64)
    left_suffix = np.cumsum(left[::-1])[::-1]
    right_prefix = np.cumsum(right)
    costs = (
        left_suffix[candidates] / left_total
        + right_prefix[candidates - 1] / right_total
    )
    return int(candidates[np.argmin(costs)])


def _energy_segment(
    energy: np.ndarray,
    start: int,
    end: int,
    heuristic: MotionHeuristic,
) -> ArmSegment:
    window = energy[start:end]
    total = _positive_total(window, "segment")
    cumulative = np.cumsum(window)
    fraction = heuristic.boundary_energy_fraction
    first = start + int(np.searchsorted(cumulative, fraction * total))
    last = start + int(np.searchsorted(cumulative, (1.0 - fraction) * total)) + 2
    return ArmSegment(
        max(start, first - heuristic.boundary_padding_frames),
        min(end, last + heuristic.boundary_padding_frames),
    )


def _keep_segments_disjoint(
    left: ArmSegment, right: ArmSegment
) -> tuple[ArmSegment, ArmSegment]:
    if left.end <= right.start:
        return left, right
    midpoint = (left.end + right.start) // 2
    return ArmSegment(left.start, midpoint), ArmSegment(midpoint, right.end)


def _cross_motion_ratios(
    left: np.ndarray, right: np.ndarray, split: int
) -> tuple[float, float]:
    return (
        float(right[:split].sum() / _positive_total(right, "right")),
        float(left[split:].sum() / _positive_total(left, "left")),
    )


def _active_mask(timeline: np.ndarray, start: int, duration: int) -> np.ndarray:
    return (timeline >= start) & (timeline < start + duration)


def _scheduled_indices(
    timeline: np.ndarray, start: int, segment: ArmSegment
) -> np.ndarray:
    local = np.clip(timeline - start, 0, segment.length - 1)
    return local + segment.start


def _round_fraction(numerator: int, denominator: int) -> int:
    return (2 * numerator + denominator) // (2 * denominator)


def _positive_total(values: np.ndarray, label: str) -> float:
    total = float(np.sum(values))
    if not np.isfinite(total) or total <= 0:
        raise ValueError(f"{label} motion energy must be finite and positive")
    return total


def _validate_vector(values: np.ndarray) -> np.ndarray:
    vector = np.asarray(values)
    if vector.ndim != 2 or vector.shape[1] != 14 or len(vector) < 2:
        raise ValueError(f"expected a [frames, 14] PiperX vector, got {vector.shape}")
    if not np.all(np.isfinite(vector)):
        raise ValueError("PiperX vector contains non-finite values")
    return vector


def _validate_heuristic(heuristic: MotionHeuristic) -> None:
    if heuristic.smoothing_frames < 1:
        raise ValueError("smoothing_frames must be positive")
    if not 0 < heuristic.search_start_fraction < heuristic.search_end_fraction < 1:
        raise ValueError("split search fractions must satisfy 0 < start < end < 1")
    if not 0 <= heuristic.boundary_energy_fraction < 0.5:
        raise ValueError("boundary_energy_fraction must be in [0, 0.5)")
    if not 0 <= heuristic.maximum_cross_motion_ratio < 1:
        raise ValueError("maximum_cross_motion_ratio must be in [0, 1)")


def detect_motion_interval(
    values: np.ndarray, arm: str, heuristic: MotionHeuristic = MotionHeuristic()
) -> ArmSegment:
    """One arm's active span without assuming cross-arm task independence."""
    _validate_heuristic(heuristic)
    energy = arm_motion_energy(values, arm, heuristic)
    return _energy_segment(energy, 0, len(energy), heuristic)
