"""Select corroborated current-frame additions without erasing existing masks."""

from ..interaction.robot_discovery import robot_entry_side


def assess_arm_reseed(before, crop_mask, full_mask, reference, side):
    if side not in (0, 1) or any(
        m.shape != before.shape for m in (crop_mask, full_mask, reference)
    ):
        raise ValueError("inconsistent arm reseed geometry")
    area = int(reference.sum())
    if area == 0:
        raise ValueError("arm reseeding requires independent source support")
    corroborated = crop_mask & full_mask
    combined = before | corroborated
    old_coverage = float((before & reference).sum() / area)
    combined_coverage = float((combined & reference).sum() / area)
    selected = (
        combined_coverage >= 0.6
        and combined_coverage >= old_coverage + 0.15
        and robot_entry_side(combined) == side
        and corroborated.sum() <= max(area * 4, before.sum() * 2)
    )
    return (combined if selected else before.copy()), {
        "before_motion_coverage": old_coverage,
        "combined_motion_coverage": combined_coverage,
        "selected_as_proposal": bool(selected),
        "new_pixels": int((corroborated & ~before).sum()) if selected else 0,
        "disagreed_pixels": int((crop_mask ^ full_mask).sum()),
    }
