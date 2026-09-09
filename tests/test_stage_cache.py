import numpy as np
from real_robot_data_retime.interaction.cache import stage_key


def test_cache_tracks_behavior_and_pixels_without_checkout_path():
    namespace = {}
    exec(
        compile("def f(x):\n return [y+1 for y in x]\n", "/checkout/a.py", "exec"),
        namespace,
    )
    first = namespace["f"]
    exec(
        compile("def f(x):\n return [y+1 for y in x]\n", "/other/b.py", "exec"),
        namespace,
    )
    second = namespace["f"]
    frames = np.zeros((2, 4, 4, 3), np.uint8)
    assert stage_key(frames, [first], ()) == stage_key(frames, [second], ())
    frames[0, 0, 0] = 1
    assert stage_key(frames, [first], ()) != stage_key(
        np.zeros_like(frames), [first], ()
    )
    exec("def f(x):\n return [y+2 for y in x]\n", namespace)
    assert stage_key(frames, [first], ()) != stage_key(frames, [namespace["f"]], ())
