import os
from pathlib import Path
import pytest
from real_robot_data_retime.interaction.video import read_video
from real_robot_data_retime.interaction.pipeline import discover_task


@pytest.mark.skipif(not os.environ.get('RETIME_SAMPLE_DIR'),reason='real-video scene integration')
def test_all_six_task_scenes_are_discovered_without_metadata():
    root=Path(os.environ['RETIME_SAMPLE_DIR'])
    for task in ['drawer','letters','workpiece']:
        for ep in [0,1]:
            frames,_=read_video(root/f'{task}_{ep}.mp4',stop=8,width=424)
            assert discover_task(frames)==task
