import numpy as np
import pytest

from real_robot_data_retime.aist_timing import *


def demo():
    q = np.zeros((240, 14))
    q[:8, [6, 13]] = np.linspace(0, 1, 8)[:, None]
    q[8:, [6, 13]] = 1
    q[65:105, 0] = np.linspace(0, 1, 40)
    q[105:, 0] = 1
    q[150:190, 7] = np.linspace(0, 1, 40)
    q[190:, 7] = 1
    return q, q.copy()


def test_independent_schedule_reduces_additional_time_and_retains_endpoints():
    q, a = demo()
    l, r, c, receipt = build_proposal(q, a, 20)
    assert len(l) < len(c) < len(q)
    assert l[0] == r[0] == 0 and l[-1] == r[-1] == 239
    assert receipt["additional_parallel_frames_saved"] > 0


def test_arm_swap_has_same_optimum():
    q, a = demo()
    l, r, c, s = build_proposal(q, a, 20)
    v = np.c_[q[:, 7:], q[:, :7]]
    ll, rr, cc, ss = build_proposal(v, v, 20)
    assert len(l) == len(ll)


def test_possible_shared_hold_uses_same_source_clock():
    q, a = demo()
    q[75:170, 6] = 0.4
    q[100:180, 13] = 0.5
    l, r, c, s = build_proposal(q, q, 20)
    for lo, hi in s["contact_hypotheses"]["protected"]:
        hit = ((l >= lo) & (l < hi)) | ((r >= lo) & (r < hi))
        np.testing.assert_array_equal(l[hit], r[hit])


def test_one_active_arm_never_claims_parallel_gain():
    q, a = demo()
    q[:, 7] = 0
    l, r, c, s = build_proposal(q, q, 20)
    np.testing.assert_array_equal(l, r)
    assert s["decision"] == "common_clock_only_one_active_arm"


def test_unresolved_constant_gripper_locks_timeline():
    q, a = demo()
    q[:, [6, 13]] = 1
    l, r, c, s = build_proposal(q, q, 20)
    np.testing.assert_array_equal(l, np.arange(len(q)))
    assert s["contact_hypotheses"]["constant_gripper_ambiguity"]


def test_rejected_projection_falls_back_without_corrupting_source():
    q, a = demo()
    l, r, c, s = build_proposal(q, a, 20, projection=lambda *x: False)
    np.testing.assert_array_equal(l, c)
    np.testing.assert_array_equal(r, c)
    assert s["decision"] == "common_clock_fallback"


def test_active_motion_cannot_be_skipped():
    q, a = demo()
    with pytest.raises(ValueError, match="Non-idle"):
        validate_map(
            q,
            a,
            np.array([0, 239]),
            np.ones(14) * 0.001,
            np.ones(14) * 0.001,
            np.zeros(len(q), bool),
        )


def test_visibility_locks_restore_a_common_clock_path():
    q, a = demo()
    l, r, c, s = build_proposal(
        q,
        a,
        20,
        projection=lambda l, r, nl, nr: l == r and nl == nr,
        shared_clock_required=np.ones(len(q), bool),
    )
    np.testing.assert_array_equal(l, c)
    np.testing.assert_array_equal(r, c)
    assert s["decision"] != "common_clock_fallback"


def test_zero_time_gain_prefers_common_clock():
    q, a = demo()
    q[:, 0] = np.linspace(0, 1, len(q))
    l, r, c, s = build_proposal(q, q, 20)
    np.testing.assert_array_equal(l, c)
    np.testing.assert_array_equal(r, c)


def test_one_arm_precision_hold_is_not_deleted():
    q, a = demo()
    q[:, 7] = 0
    q[90:200, 6] = 0.4
    l, r, c, s = build_proposal(q, q, 20)
    assert s["contact_hypotheses"]["single_arm_hold_guard"]
    assert set(range(90, 200)) <= set(c)
