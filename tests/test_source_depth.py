import json
from zipfile import ZipFile

import cv2
import numpy as np
import pyarrow as pa

from real_robot_data_retime.compositing.depth import source_depth
from real_robot_data_retime import trim as trim_module


def test_raw_depth_uses_trim_offset_and_rgb_registration(tmp_path, monkeypatch):
    (tmp_path / 'meta').mkdir()
    (tmp_path / 'meta/info.json').write_text(json.dumps({}))
    references = []
    with ZipFile(tmp_path / 'depth.zip', 'w') as archive:
        for i in range(4):
            frame = np.zeros((4, 6), np.uint16)
            frame[1, 2] = 1000 + i
            ok, encoded = cv2.imencode('.png', frame)
            assert ok
            archive.writestr(f'{i}.png', encoded.tobytes())
            references.append(dict(path=f'zip://depth.zip#{i}.png'))
    table = pa.table({'observation.depth.top': references})
    monkeypatch.setattr(trim_module, 'source_episodes', lambda _: [dict(episode_index=0)])
    monkeypatch.setattr(trim_module, 'read_episode', lambda *args: table)
    transforms = np.repeat(np.eye(3)[None], 2, axis=0)
    transforms[:, 0, 2] = 1
    depth = source_depth(tmp_path, 0, dict(start=1, stop=3), transforms, (4, 6))
    try:
        assert depth(0)[1, 3] == 1001
        assert depth(1)[1, 3] == 1002
        assert depth(0)[1, 2] == 0
    finally:
        depth.close()
    assert not depth.archives and not depth.cache
