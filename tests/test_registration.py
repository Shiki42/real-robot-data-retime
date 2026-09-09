import cv2
import numpy as np
from real_robot_data_retime.interaction.registration import stabilize


def test_camera_translation_removed_without_absorbing_independent_object():
    rng = np.random.default_rng(8)
    first = rng.integers(0, 256, (160, 240, 3), dtype=np.uint8)
    frames = [first]
    for i in range(1, 5):
        f = cv2.warpAffine(
            first,
            np.array([[1, 0, i], [0, 1, 0]], float),
            (240, 160),
            borderMode=cv2.BORDER_REFLECT,
        )
        cv2.rectangle(f, (50 + i * 5, 90), (75 + i * 5, 115), (0, 0, 0), -1)
        frames.append(f)
    aligned, transforms, confidence = stabilize(np.array(frames))
    assert confidence[1:].min() > 0.5
    assert np.allclose(transforms[-1, :2, 2], [-4, 0], atol=0.2)
    assert (
        np.mean(np.abs(aligned[-1, 10:50, 10:200].astype(float) - first[10:50, 10:200]))
        < 5
    )
