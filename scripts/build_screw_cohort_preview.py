"""Build validated per-episode previews and a shared episode selector."""

import argparse
import json
from pathlib import Path

from build_screw_preview import build


def build_cohort(root):
    folders = sorted(root.glob("ep[0-9][0-9][0-9]"))
    if not folders:
        raise ValueError("no episode artifacts")
    links = []
    for folder in folders:
        build(folder)
        cases = json.loads((folder / "cases.json").read_text())
        links.append(
            f'<a href="{folder.name}/">Episode {int(folder.name[2:])} · {len(cases)} 种时序</a>'
        )
    (root / "index.html").write_text(
        """<!doctype html><html lang="zh"><meta charset="utf-8">
<title>插螺丝 · 多 episode 检查</title><style>
body{background:#11161c;color:#eef2f6;font:18px system-ui;margin:48px auto;max-width:900px;padding:24px}
a{display:block;background:#202a35;color:#9fd4ff;padding:20px;margin:12px 0;border-radius:12px;text-decoration:none}p{line-height:1.8;color:#bcc7d4}
</style><h1>插螺丝 · 多 episode 检查</h1>
<p>每条示范提供左臂先行、中间时序、右臂先行三个版本。准备末尾的微调连续衔接，插入后的右臂撤回保持原速。</p>
<p>Episode 1 是原第 709 帧问题的修复版本。可在每条视频下逐帧检查，或跳转到各轮插入和撤回。</p>
"""
        + "".join(links)
        + "</html>"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    build_cohort(parser.parse_args().root)
