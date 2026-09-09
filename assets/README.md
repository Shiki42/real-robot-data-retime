# PiperX geometry snapshot

`piper_x_description.urdf` is copied from the user-specified RoboVisualize local
checkout on 2026-09-09. It includes the uncommitted gripper_base_joint rotation
`rpy="0 0 1.5708"`. Other local RoboVisualize changes were left untouched.

Mesh filenames resolve against the assets directory supplied by the installed
RoboVisualize checkout. Robot FK is reused from `PiperXModel` in that project.
Both bases face +X and are placed at Y=+0.245 m and Y=-0.245 m, respectively,
matching RoboVisualize's `DualPiperXBackend` default.
