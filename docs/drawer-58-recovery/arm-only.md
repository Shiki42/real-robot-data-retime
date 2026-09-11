# 实测宽度、仅机械臂检查：87源全量预检

2026-09-12，按用户明确提供的信息和授权重新运行：抽屉宽17.8cm、深20cm；
物体边长3cm且夹持在两指中间，不再执行物体球与抽屉检查；二维投影交集不拒绝。
高度仍为未实测的7cm。保留已有机械臂/夹爪网格对完整扫掠体积的候选检查，
以及减速和等待期间对瞬时抽屉体积的检查，保留原间隙和连续采样余量。

**87/87源的两个变体均通过，174份计划，0源拒绝。** 上轮69源通过、18源拒绝。
源60沿用已记录的approach_start=196人工修正，人工额度仍为1/5。
没有新增人工标注，没有更改2mm峰值容差、A/B/C定义、相位差0.5或平滑时长。

[完整结果](arm-only-preflight.json) · [配置](arm-only-config.json) · [精确补丁](arm-only.patch)

独立实验根：`/home/coder/share/drawer-178mm-arm-only-20260912`。
使用该目录runtime/src和已有robo-visualize/src作为PYTHONPATH，3进程独立处理各源。
[scripts/arm-only-preflight.py](scripts/arm-only-preflight.py) 保存执行脚本。
配置保留旧held_object_radius_m字段，但本实验uniform路线已不读取它进行碰撞拒绝。
补丁针对实验分支源码，供复现此预检，不是渲染路线已经集成完成的声明。

验证了87个唯一源、174份计划的当前producer、实际宽度、阶段验证、减速几何验证、
两相位差0.5及左右映射单调性。逐源planned.json和两份plan/mapping均保存在work下。
没有渲染视频、没有finalize，故不能把174份计划称作174条已验收视频。
现有渲染端的遮挡合成和验证还需相应实现；本次未改生产源码和main，也未覆盖旧证据。
