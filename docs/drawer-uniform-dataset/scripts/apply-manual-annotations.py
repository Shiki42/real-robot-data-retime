"""Reproduce source35/61 annotations from the recorded original analysis caches."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--source-config', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
config = json.loads(args.source_config.read_text())
args.output.mkdir(parents=True, exist_ok=False)
notes = Path(__file__).resolve().parents[1]
for ep in [35, 61]:
    note = json.loads((notes / f'manual-{ep:03d}.json').read_text())
    source = Path(config['analyses'][str(ep)])
    target = args.output / f'episode_{ep:03d}'
    identity = source / ('interaction_timeline.json' if ep == 35 else 'segmentation.npz')
    digest_key = 'source_timeline_sha256' if ep == 35 else 'source_segmentation_sha256'
    if hashlib.sha256(identity.read_bytes()).hexdigest() != note[digest_key]:
        raise ValueError(f'source {ep} is not the reviewed original analysis')
    shutil.copytree(source, target)
    if ep == 35:
        path = target / 'interaction_timeline.json'
        timeline = json.loads(path.read_text())
        assert timeline['drawer_motion']['close_start'] == note['old']
        timeline['drawer_motion']['close_start'] = note['new']
        timeline['drawer_motion']['manual_annotation'] = note
        path.write_text(json.dumps(timeline, indent=2))
    else:
        path = target / 'segmentation.npz'
        with np.load(path) as archive:
            values = {key: archive[key] for key in archive.files}
        assert values['frame_shape'].tolist() == [362, 640]
        robots = np.unpackbits(values['robots'], axis=-1, count=640)
        region = np.zeros((362, 640), np.uint8)
        cv2.fillPoly(region, [np.array(note['change']['polygon_xy'], np.int32)], 1)
        start, stop = note['change']['frames_half_open']
        robots[start:stop, 1][:, region.astype(bool)] = 0
        values['robots'] = np.packbits(robots, axis=-1)
        np.savez_compressed(path, **values)
    (target / 'manual-annotation.json').write_text(json.dumps(note, indent=2))
    config['analyses'][str(ep)] = str(target.resolve())
(args.output / 'config.json').write_text(json.dumps(config, indent=2))
