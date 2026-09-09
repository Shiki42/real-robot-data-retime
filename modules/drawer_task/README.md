# 拉抽屉任务（已认可版本）

本模块固定用户认可的 `drawer_0` 双臂同步视频版本：462 帧，30 FPS，15.4 秒。
这是仓库内的独立任务模块，不是指向另一个仓库的 Git submodule。
主仓库 `src/` 的实验路线和入口保持原样。本模块不导入它们。

## 固定内容

- `analysis/src/`：当时实际使用的 `runtime-photometric-final` 分析快照。
- `render/src/`：当时实际使用的 `runtime-render-safe-color` 渲染快照。
- `render/checkpoint.py`：来自提交 `6d7460d` 的带来源校验的渲染入口。
- `evidence/`：原始分析报告、渲染报告和逐帧来源映射。
- `version.json`：认可视频与输入的 SHA-256、服务器上的原始产物位置。
- `requirements.txt`：冻结时运行环境的直接依赖版本和 CoTracker 提交。
- `freeze-manifest.json`：固定文件哈希。每次运行前自动校验。

两份快照刻意独立保存：分析与渲染当时使用的实现有差异，不能用当前
实验分支代替。原始 Python 文件逐字保留，包括快照内部复用的其他任务
辅助代码；入口只开放抽屉任务。后续修改应建立新版本，勿覆盖此版本。

## 使用

依赖清单面向 Linux CUDA 12.8 运行端；使用 Python 3.11+、FFmpeg。
完整分析需要 GPU 和模型权重。建议独立虚拟环境：

```bash
python -m venv .venv-drawer
.venv-drawer/bin/pip install -r modules/drawer_task/requirements.txt
.venv-drawer/bin/python modules/drawer_task/run.py verify
.venv-drawer/bin/python modules/drawer_task/run.py analyze \
  --input /absolute/path/drawer.mp4 --output /absolute/path/new-analysis
.venv-drawer/bin/python modules/drawer_task/run.py render \
  --input /absolute/path/drawer.mp4 --analysis /absolute/path/new-analysis \
  --output /absolute/path/new-render
```

启动器使用 Python isolated 模式，忽略外部 `PYTHONPATH` 和当前工作目录，
防止载入主分支正在变化的同名 Python 包。输出目录必须是新目录。
模型版本见两份快照的 `segmentation/model_versions.py`。

## 重现已经认可的成片

在 Coder A 上，复用原始完整分析检查点，可跳过神经网络重新推理：

```bash
/home/coder/share/retime-model-experiments-20260910/.venv/bin/python \
  modules/drawer_task/run.py render \
  --input /home/coder/share/retime-accuracy-20260910/drawer_0-render-safe-color/source.mp4 \
  --analysis /home/coder/share/retime-accuracy-20260910/drawer_0-final-photometric \
  --output /absolute/path/new-drawer-render
```

原视频、模型权重和大型分析数组不放入 Git；原始位置及哈希在版本记录中。
当时分析复用了已有测量，因此从原视频全新推理不承诺逐像素重现。
固定分析快照也保留了当时 CoTracker 的显存需求，建议单任务运行。

用户已认可这段视频的视觉效果；历史报告的 `pending` 状态原样保存，
不追改历史证据。这不表示其他输入视频都通过验收，也不是机械臂实体执行
安全认证：本次视频规划仅检查投影轮廓，没有关节空间碰撞验证。

## 迁移验证

独立入口重新生成的 MP4 与用户认可视频 SHA-256 完全相同，逐帧来源映射
文件也完全相同。结果见 `evidence/migration-verification.json`。
主分支测试：88 项通过、5 项跳过。
