# 插螺丝：最后就位后只等待一次

数据集 `Shiki42/piperx-screw-0910-25ep-raw`，revision
`824b6382ef4858213e6ea42ea61104110d911448`。
本轮重做 episode 0、1、2、12、24，每条 u=0.15、0.50、0.85，共 15 条。
新版工作目录 `/home/coder/share/screw-retiming-20260912/final-ready`（coder a），
预览 `http://127.0.0.1:38779/final-ready/`。旧 `cohort/` 保留作比较。

## 调整完成与同步边界

旧版用宽松 TCP 就位容差提前宣布左臂到位，又把其后剩余源帧称为连续微调，
导致原片中的「中途等待→最后调整」仍被带入输出。源时间前进不能证明机械臂在动。

现在以人工核查的协同入口姿态为参考，寻找左臂最后一次调整的结束：从候选帧到协同入口，
所有 measured state 和 commanded action 的关节偏差都不超过 0.05 度、开度偏差不超过 0.1 mm。
必须整个后缀满足条件，不能把中间的暂时停顿当作终点。`final_left_pose` 负责此判断，
`final_left_pose_evidence` 保存逐通道差异和参考帧。

左臂先完成全部接触前调整，到最终插入姿态后才允许减速等待。等待后不再执行额外调整，
仅将已核查为静止的源时间尾段并入原始同步入口。右臂到位较早时，复用原有减速/启动插帧，
让启动后的剩余接近段与左臂动作同时结束。每轮最多一臂被安排等待。

左臂未出现可确认的稳定终点（检测结果等于同步入口）时，不强行添加停车；
通过连续减速消化提前量。此时部分左臂先行样例会连续接近，而不是提前固定在一个未经确认的姿态。
不再使用旧版宽松 readiness 与三倍速残余对齐；两臂共用原来的有界静止压缩和视频/动作插值。
准备段静止压缩至少 0.2 秒、两端各保留 1 帧，避免每处原始等待都留下 0.4 秒尾巴。

### ep002 的边界

| 轮次 | 左臂最后就位源帧 | 同步入口源帧 |
| --- | ---: | ---: |
| 1 | 327 | 345 |
| 2 | 805 | 807 |
| 3 | 1171 | 1282 |
| 4 | 1715 | 1740 |
| 5 | 2263 | 2285 |

这里两个源帧不同并不意味着输出要走完整段等待：中间经验证不再含左臂调整的尾段被压缩，
最终仍在相同的原始双臂帧进入同步阶段。
第四轮原来的同步入口是 1785，但 1740 之后已有首次接近/尝试、退开和再调整；
因此将保护入口前移至 1740。接触可能已发生的调整不能独立提前，这部分保留原始双臂同步。
所有轮次的右臂插入后撤回仍是上一轮的强制原速后缀。

## 重现与检查

```sh
PYTHONPATH=src python scripts/analyze_screw_readiness.py \
 --source /home/coder/share/screw-retiming-20260912/raw \
 --config docs/screw-cohort/ep002.json
PYTHONPATH=src python -m real_robot_data_retime.screw \
 --source /home/coder/share/screw-retiming-20260912/raw \
 --work /home/coder/share/screw-retiming-20260912/final-ready/ep002 \
 --config docs/screw-cohort/ep002.json --prepare
python scripts/build_screw_cohort_preview.py /home/coder/share/screw-retiming-20260912/final-ready
```

其余配置为 ep000.json、ep012.json、ep024.json 和 ../screw-pilot/config.json。
`verify_screw_cohort.py` 重算源时间并逐帧与生成映射核对，同时检查 action/state 与四条视频时间戳；
汇总见 validation.json。44 项相关测试覆盖最后调整判定、每轮仅一臂等待、等待后无左臂调整、
右臂强制撤回及不合法改动负例，并使用原始 episode 1、2 的动作数据做回归。

主视角插入与最终收纳每条抽查 30 帧；腕视角逐帧检查。视觉检查仍是抽样，
背景亮度接缝和前景边缘残影等合成限制未因此消失。没有把这 5 条示范的结果泛化为全部 25 条。
