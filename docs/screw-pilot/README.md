# 插螺丝：连续进入插入，右臂固定撤回

数据：`Shiki42/piperx-screw-0910-25ep-raw`，revision
`824b6382ef4858213e6ea42ea61104110d911448`，episode 1。

## 调度规则

左臂的「收纳上一件＋拾取下一套筒＋靠近」保持连续，先到的一臂可以在就位点等待。
晚到的一臂持续执行原轨迹，直接进入插入，不再强制经历一次减速到零和重新启动。
提前到达的一臂按另一臂的预计完成时间启动，双方同时接入原始协同区间。

等待有足够时长时，复用 `lift_clock` 的 0.5 秒减速、0.3 秒启动。
时间差较小时，在最后接近段复用 `speed_ramp` 构造速度始终为正的平滑变速，
消化小幅提前量，避免为了同步而完整刹停。
`stage_delays` 在名义拾取至就位时长上均匀采样相对开始时间。

每次实际插入完成后，右臂向右下方撤回到等待姿态，是这次插入必须连续执行的后缀。
右臂从协同开始至撤回结束逐帧保持原始源时间，期间不能等待、插帧或重新排序。
下一轮右臂拾取的开始时间必须晚于该撤回终点。
左臂可以同时开始其收纳/下一次拾取，或在整个动作开始前等待，内部不插入暂停。

| 轮次 | 协同开始 | 双臂协同结束 | 右臂任务结束（撤回到位） |
| --- | ---: | ---: | ---: |
| 1 | 365 | 435 | 469 |
| 2 | 805 | 895 | 989 |
| 3 | 1310 | 1370 | 1427 |
| 4 | 1785 | 1985 | 2063 |
| 5 | 2405 | 2460 | 2607 |

以上为原始 episode 帧号，含两端。最后一轮保留全部可用原始尾段，直到 episode 结束。
`coupled.right_retreat` 明确记录右臂插入任务的后缀；
`mandatory_previous_right_retreat` 在其与左臂独立动作重叠的时段落实同一约束。

## 同步与视频

只通过既有 `compress_static_spans` 压缩被 state 和 action 共同证实的有界静止。
`sample_rows` 插值 action/state；`FlowFrames` 插值整臂及携带物前景，
双腕视频通过既有 `remap_video` 使用相同的连续源时间。
实际协同插入保持原始同步整帧，强制右臂撤回保持原速，末次收纳保持原样。

主视角为 `observation.images.right_environment_1`。沿用已核查的整臂和携带螺丝分割，
套筒区/收纳盒跟随左时钟、螺丝区跟随右时钟；螺丝头及螺杆也参与前景移除和插帧。
模型分割来自 `approach-v2`，此轮只改变调度与预览。
边界和提示完整记录于 `config.json`，仍是人工核查的单条 episode 样例。

## 产物

工作目录：`/home/coder/share/screw-retiming-20260912/flow-v3`（coder a）。
严格裁剪后的原片为 2604 帧 / 86.8 秒；只删开头 4 帧，不补造终止静止。

| 样例 | u | 帧数 | 秒 |
| --- | ---: | ---: | ---: |
| 左臂先行 | 0.15 | 3209 | 106.967 |
| 中间时序 | 0.50 | 2473 | 82.433 |
| 右臂先行 | 0.85 | 3425 | 114.167 |

```bash
PYTHONPATH=src python -m real_robot_data_retime.screw \
  --source /home/coder/share/screw-retiming-20260912/raw \
  --work /home/coder/share/screw-retiming-20260912/flow-v3 \
  --config docs/screw-pilot/config.json --prepare
python scripts/build_screw_preview.py /home/coder/share/screw-retiming-20260912/flow-v3
```

省略 `--prepare` 可以重用经来源和配置核对的掩码；`--prepare-only` 只做分割。
每条包含主视角 MP4/WebM、左右腕 MP4、`trajectories.parquet`、
`source_mapping.npz` 和 `report.json`。旧版视频保留在上级目录及 `approach-v2`。
预览使用视频实际 PTS 做逐帧跳转，并提供每轮插入、撤回、撤回完成的入口。

## 验证

- 35 项相关测试通过，覆盖晚到臂连续推进、短时间差不停车、右臂强制撤回及违规负例。
- 所有 action/state 与连续源时间映射一致；所有视频帧数和 PTS 与运动时间轴一致。
- 双腕视频逐帧像素校验；每条抽查 30 个同步保护段主视角帧。
- 每轮右臂从插入到撤回结束全段严格逐帧匹配原轨迹，下一次拾取不得提前。
- 插入前晚到臂不被调度器停住；全程无新加双臂同时等待。
- 背景恢复缺失像素数均为 0。

数值收据见 `validation.json`；插入前源时间推进的前后对照见 `entry-continuity.json`。
撤回终点经原始动作静止信号与主视角共同核查。调度沿用用户给定的独立工作区假设。
