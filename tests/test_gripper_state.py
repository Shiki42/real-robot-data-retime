import numpy as np
from real_robot_data_retime.interaction.gripper_state import aperture_states


def test_silhouette_states_do_not_claim_missing_frames_are_closed():
    aperture = np.r_[
        np.full(10, 20),
        np.linspace(20, 2, 10),
        np.full(10, 2),
        np.linspace(2, 20, 10),
        np.full(10, 20),
    ].astype(float)
    aperture[0] = np.nan
    states = aperture_states(aperture, 30)
    assert states[0] == "UNKNOWN"
    assert states[5] == "OPEN" and states[25] == "CLOSED"
    assert "CLOSING" in states[10:20] and "OPENING" in states[30:40]
