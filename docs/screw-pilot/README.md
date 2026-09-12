# 插螺丝：五轮时序重组样例

源数据为 `Shiki42/piperx-screw-0910-25ep-raw`，固定 revision
`824b6382ef4858213e6ea42ea61104110d911448`。本次取 episode 1，
在 coder a 生成三条完整五轮视频，供检查动作分段与视觉合成。

每轮的独立段采用已有 `stage_delays` 均匀相对起始时序：首轮左手拿套筒，
后四轮左手连续收纳上一件并拿下一套筒；右手撤回、拾取下一颗螺丝。
独立段内部不添加暂停，不跨轮交换物件。只通过已有 `compress_static_spans`
移除 state 和 action 全区间范围均不超过 0.3° / 0.5 mm 的静止帧。
动作帧保持原始速度和次序，等待时重复对应边界姿态。

五次插入的左右时钟严格相同、逐帧重放原始整帧；最后收纳也保持原样。
同步保护段包括接近和释放/撤回边界。原始 episode 帧号（含两端）：

| 插入 | 开始 | 结束 |
| --- | ---: | ---: |
| 1 | 300 | 435 |
| 2 | 760 | 895 |
| 3 | 1250 | 1370 |
| 4 | 1770 | 1985 |
| 5 | 2340 | 2460 |

复用原有严格静止裁剪：删除开头 4 帧，保留原始结尾；不补造终止静止。
裁剪后原片 2604 帧 / 86.8 秒。不同偏移允许更长的单臂等待，目标是时序多样性。

| 样例 | u | 帧数 | 秒 |
| --- | ---: | ---: | ---: |
| 左臂先行 | 0.15 | 3281 | 109.367 |
| 中间时序 | 0.50 | 2462 | 82.067 |
| 右臂先行 | 0.85 | 3280 | 109.333 |

主视角使用 `observation.images.right_environment_1`。整臂分割复用 `SamVideo`；
背景曝光配准、真实像素背景恢复、视频重映射、轨迹导出和像素校验均复用现有实现。
收纳盒和套筒工作区跟随左源时钟，螺丝工作区跟随右源时钟；整臂前景可以跨越
工作区分界。被遮挡而不与画面入口相连的夹爪片段仍被保留。
右臂在第五轮开头的撤回轨迹另行播种，避免其离开画面时跟踪丢失。

`config.json` 完整记录本次人工核查的分段、像素区域和 SAM 提示。
这是可复现的样例入口，不表示已经完成全部 25 条数据的自动语义分段。
按用户给定的工作区独立假设调度，未新增碰撞搜索。数据没有深度，投影重叠沿用
现有 RGB 路径的右臂前景顺序。

## 复现

在已经安装仓库依赖和分割模型的远端运行：

```bash
PYTHONPATH=src python -m real_robot_data_retime.screw \
  --source /home/coder/share/screw-retiming-20260912/raw \
  --work /home/coder/share/screw-retiming-20260912 \
  --config docs/screw-pilot/config.json --prepare

python scripts/build_screw_preview.py /home/coder/share/screw-retiming-20260912
python scripts/serve_action_preview.py --port 38770 \
  --directory /home/coder/share/screw-retiming-20260912
```

`--prepare-only` 仅生成分割；不带 `--prepare` 则使用来源及配置校验通过的分割。
重新生成会覆盖同名样例。每条包含主视角 MP4/WebM、左右腕 MP4、
`trajectories.parquet`、`source_mapping.npz` 和 `report.json`。
原始对照视频 `original.webm` 是本次额外导出的裁剪原片。

## 验证

- 三条视频、双腕视频、action/state 的所有帧数及 PTS 一致。
- 全部 action/state 逐值匹配同一组左右源时间。
- 五个同步区间与末次收纳逐帧保留；独立段没有新加双臂同时等待或段内暂停。
- 每条抽查双腕各 40 帧，保护段主视角 30 帧；最大像素 MAE 分别为 3.404、2.515
  （0–255 标度，编码容差为 6）。
- 三条所有阶段的背景修复缺失像素数均为 0。
- 视觉检查涵盖五轮同时运动的源帧/合成帧对照、插入边界和最终收纳；浏览器播放、
  跳转和逐帧控制另行检查。

详细数值见 `validation.json`。视频位于 coder a 的上述 work 目录，不上传或改写源数据集。
