# 插螺丝多 episode 验证

数据集与 revision 同 `../screw-pilot/config.json`。选择 episode 0、1、2、12、24，
覆盖开头、中间和末尾示范；每条生成 u=0.15、0.50、0.85，共 15 条完整视频。
每条包含 5 次插入，合计检查 75 个插入衔接和右臂撤回区间。

## 修复

旧调度把左臂已经就位后的微调/静止时长也计入准备时长，导致右臂错误刹停。
现在通过 measured state 和 commanded action 的 PiperX 正运动学分别判断持续就位：
测量 TCP 位置 5 mm、方向 2°；命令位置 10 mm、方向 3°；夹爪开度 0.5 mm。
必须从候选帧一直到参考帧都满足条件，不能仅凭瞬时接近判断。
这些阈值用于当前示范的就位判定，仍需人工确认任务边界。

准备后的微调沿原轨迹做正速度重映射；相对于压缩静止后的原时钟最多 3 倍速，
用已有 speed_ramp/sample_rows 衔接。没有删掉微调路径，也不改变真实插入区间。
只在确实能形成至少 0.3 秒等待时才生成完整停车，采用原有 0.5 秒减速和 0.3 秒启动。
右臂等待不得延续到左臂物理准备完成之后；插入及强制撤回逐帧保持原速。

Episode 1 的中间时序，第 2 轮不再有额外右臂 hold；新视频在输出 707 帧进入
源帧 805 的同步插入，709 帧两臂均对应源帧 807。视频时长改变，因此旧帧号与新帧号
不代表相同动作阶段。小型原始 state/action 回归数据保存在 tests/fixtures/screw_ep001.npz，
来源为上述 HF revision 的 episode 1，未包含视频。

## 重现

在 coder a 的 `/home/coder/share/real-robot-data-retime-screw` 运行：

```sh
PYTHONPATH=src python -m real_robot_data_retime.screw \
 --source /home/coder/share/screw-retiming-20260912/raw \
 --work /home/coder/share/screw-retiming-20260912/cohort/ep000 \
 --config docs/screw-cohort/ep000.json --prepare
python scripts/build_screw_cohort_preview.py /home/coder/share/screw-retiming-20260912/cohort
```

其余配置为 ep002.json、ep012.json、ep024.json 和 ../screw-pilot/config.json。
配置记录每轮人工接触边界、右臂撤回、分割提示和持续就位证据。
`scripts/analyze_screw_readiness.py` 可从原始 state/action 和 PiperX URDF 重新计算就位帧。
分割复用既有 SAM 视频传播；新增样本修正了高举右臂时背景误选及接近时夹爪漏分割。
细长携带物允许独立提示，必要时从其稍后可见帧反向传播到拾取点。

## 验证范围

validation.json 汇总逐条运动映射、视频帧数/PTS、双腕逐帧像素误差及保护段检查。
主视角同步保护段每条抽查 30 帧；非同步段另外查看每轮插入前和撤回附近画面。
这属于 5 条示范的回归检查，不等于所有 25 条已完成自动标注，也不保证分割每个像素无误。

预览：`http://127.0.0.1:38779/cohort/`。原始 flow-v3 保留用于前后对照。

画面仍存在少量背景亮度接缝（尤其 episode 24）及前景边缘残影；这些属于合成画质限制，不能把数值同步通过等同于像素级无瑕疵。
