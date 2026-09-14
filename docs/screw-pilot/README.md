# 插螺丝 episode 1 回归样例

当前调度、算法和复现说明见 [多 episode 验证](../screw-cohort/README.md)。
配置 `config.json` 记录 episode 1 的原始同步区间、最终左臂就位点、右臂等待点、
固定撤回后缀及分割提示。数据 revision 保存在配置内。

新版产物：`/home/coder/share/screw-retiming-20260912/final-ready/ep001`（coder a）。
每条包含主视角 MP4/WebM、左右腕 MP4、trajectories.parquet、source_mapping.npz 和 report.json。
旧版 `flow-v3/`、`cohort/ep001/` 保留作比较；新的时长与帧号已经改变。

当前左臂终点取最后调整完成后的姿态。无法在同步边界前确认稳定终点时，不强行停车；
早到臂连续减速衔接。每轮最多一臂等待，右臂插入到撤回结束保持原始逐帧时钟。
验证收据见 validation.json，与其他 episode 汇总在 ../screw-cohort/validation.json。
entry-continuity.json 记录本轮插入入口与等待者，不再用旧视频输出帧号定位新版动作。
