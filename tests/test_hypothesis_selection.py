from real_robot_data_retime.timeline.hypotheses import choose_episodes


def test_joint_selection_avoids_greedy_long_false_episode():
    def event(obj, start, end, score):
        return dict(
            object_id=obj,
            robot_id="left",
            grasp_start=start,
            pickup_frame=start + 1,
            release_frame=end,
            score=score,
        )

    long = event(0, 5, 80, 0.95)
    correct = event(0, 5, 20, 0.85)
    second = event(1, 40, 60, 0.9)
    result = choose_episodes([long, correct, second], 2, "letters", 30)
    assert result == [correct, second]


def test_one_object_cannot_be_assigned_to_both_arms():
    candidates = [
        dict(
            object_id=0,
            robot_id=side,
            grasp_start=1,
            pickup_frame=2,
            release_frame=10,
            score=score,
        )
        for side, score in [("left", 0.8), ("right", 0.9)]
    ]
    assert len(choose_episodes(candidates, 1, "letters", 30)) == 1
