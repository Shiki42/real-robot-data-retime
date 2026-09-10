import numpy as np

from real_robot_data_retime.segmentation.sam3_video import (
    bind_root_ids,
    track_two_roots,
)


class Fake:
    def __init__(self):
        self.sessions = {}
        self.created = 0

    def output(self, sid, t):
        ids = np.array([sid * 10 + 2, sid * 10 + 7])
        m = np.zeros((2, 4, 8), bool)
        m[0, :, :2] = True
        m[1, :, -2:] = True
        return {
            "frame_index": t,
            "outputs": {"out_obj_ids": ids, "out_binary_masks": m},
        }

    def handle_request(self, r):
        if r["type"] == "start_session":
            self.created += 1
            self.sessions[self.created] = False
            return {"session_id": self.created}
        if r["type"] == "add_prompt":
            return self.output(r["session_id"], r["frame_index"])
        if r["type"] == "close_session":
            return {}

    def handle_stream_request(self, r):
        sid = r["session_id"]
        assert not self.sessions[sid], "Direction state reused"
        self.sessions[sid] = True
        for t in (
            range(2, 6) if r["propagation_direction"] == "forward" else range(1, -1, -1)
        ):
            yield self.output(sid, t)


def test_independent_sessions_and_bindings():
    p = Fake()
    m, v, seen, r = track_two_roots(p, "unused", 2, 6, 4, 8)
    assert p.created == 2 and v.all() and seen.all()
    assert r[0]["id_mapping"] != r[1]["id_mapping"]
    assert all(not x["unknown_ids"] for x in r)


def test_assignment_not_id_order():
    m = np.zeros((2, 4, 8), bool)
    m[1, :, :2] = True
    m[0, :, -2:] = True
    assert bind_root_ids([5, 9], m) == {9: 0, 5: 1}
