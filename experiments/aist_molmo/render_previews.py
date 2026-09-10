"""Pinned Molmo pilot. Experimental; no physical-success claim."""

import hashlib
import json
import time
from functools import lru_cache

import cv2
import numpy as np
from _common import run_stage, work_dir, write_video


def dilate(m, k=5):
    return cv2.dilate(m.astype(np.uint8), np.ones((k, k), np.uint8)).astype(bool)


def label(im, text, xy, color=(235, 240, 248), size=0.6):
    cv2.putText(im, text, xy, cv2.FONT_HERSHEY_SIMPLEX, size, color, 1, cv2.LINE_AA)


def main(args):
    """Run the pinned Molmo pilot stage; see README.md for input contracts."""
    P = work_dir()
    root = P / "experiments/molmo_video_v4"
    episodes = [args.episode] if args.episode else [15, 8]
    for ep in episodes:
        d = root / f"ep{ep}"
        deadline = time.time() + 1200
        while not (d / "robots.npz").exists():
            if time.time() > deadline:
                raise RuntimeError("Tracking did not finish")
            time.sleep(5)
        plans = json.loads(
            (d / ("screened_plans.json" if args.screened else "plans.json")).read_text()
        )
        cap = cv2.VideoCapture(str(P / f"data/molmo_preview/ep{ep}/source.mp4"))
        frames = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        cap.release()
        frames = np.stack(frames)
        n, h, w, _ = frames.shape
        z = np.load(d / "robots.npz")
        packed = z["robots"]
        valid = z["valid"]
        if len(frames) != plans[0]["source_frames"]:
            raise RuntimeError("Numeric/video frame mismatch")

        @lru_cache(maxsize=128)
        def masks(t):
            return np.unpackbits(packed[t], axis=-1, count=w).astype(bool)

        anchor = plans[0]["left_stage_reference_frame"]
        reference = frames[anchor]
        refarms = masks(anchor)
        left_rest = dilate(refarms[0])
        right_rest = dilate(refarms[1], 9)
        right_objects = np.max(cv2.absdiff(frames[0], reference), axis=2) > 28
        right_objects &= ~dilate(masks(0).any(0) | refarms.any(0), 11)
        right_objects = dilate(
            cv2.morphologyEx(
                right_objects.astype(np.uint8),
                cv2.MORPH_OPEN,
                np.ones((3, 3), np.uint8),
            ).astype(bool)
        )

        @lru_cache(maxsize=48)
        def left_layer(t):
            if t < anchor:
                m = dilate(masks(t)[0] | refarms[0])
                return (m, m)
            diff = np.max(cv2.absdiff(frames[t], reference), axis=2)
            xx = np.indices(diff.shape)[1]
            m = (diff > 20) & ~right_rest & (xx < 384)
            m = cv2.morphologyEx(
                m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
            )
            m = dilate(
                cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)).astype(
                    bool
                ),
                3,
            )
            arm = dilate(masks(t)[0] | refarms[0], 3)
            foreign = dilate(masks(t)[1] | refarms[1], 21)
            m = m & ~foreign & ~right_objects | arm
            opaque = arm | (diff > 45) & ~right_objects & m
            return (m, opaque)

        def composite(l, r):
            if l == r:
                return (frames[l].copy(), 0, True)
            if not (valid[l, 0] and valid[r, 1]):
                return (frames[r].copy(), -1, False)
            base = frames[min(r, anchor)].copy()
            if r > anchor:
                rp = dilate(masks(r)[1] | refarms[1])
                base[rp] = frames[r][rp]
            m, opaque = left_layer(l)
            if l >= anchor:
                soft = m & ~opaque
                base[soft] = np.clip(
                    base[soft].astype(np.int16)
                    + frames[l][soft].astype(np.int16)
                    - reference[soft].astype(np.int16),
                    0,
                    255,
                )
            base[opaque] = frames[l][opaque]
            overlap = dilate(masks(l)[0], 3) & dilate(masks(r)[1], 3)
            overlap[int(h * 0.9) :] = False
            return (base, int(overlap.sum()), True)

        receipts = []
        for plan in plans:
            name = plan["name"]
            maps = np.load(d / (name + ".npz"))
            left, right = (maps["left"], maps["right"])
            outn = len(left)
            total = max(n, outn)
            fps = plan["fps"]
            target = d / (name + "_comparison.mp4")
            counts = []
            missing = []
            tiles = []
            sample = set(np.linspace(0, outn - 1, 12).astype(int))
            last = None

            def generate():
                for k in range(total):
                    kk = min(k, outn - 1)
                    l, r = (int(left[kk]), int(right[kk]))
                    img, overlap, ok = composite(l, r)
                    if k < outn:
                        counts.append(overlap)
                        missing.append(not ok)
                        if k in sample:
                            tile = np.zeros((204, 320, 3), np.uint8)
                            tile[24:] = cv2.resize(img, (320, 180))
                            label(
                                tile,
                                f"{k / fps:.1f}s L{l / fps:.1f} R{r / fps:.1f}",
                                (4, 17),
                                size=0.4,
                            )
                            tiles.append(tile)
                    canvas = np.full((624, 1280, 3), (15, 22, 32), np.uint8)
                    canvas[48:408, :640] = frames[min(k, n - 1)]
                    canvas[48:408, 640:] = img
                    label(
                        canvas,
                        f"ORIGINAL  {min(k, n - 1) / fps:.2f}s / {n / fps:.2f}s",
                        (18, 31),
                        size=0.73,
                    )
                    label(
                        canvas,
                        "COUNTERFACTUAL PREVIEW"
                        + ("  [END HOLD]" if k >= outn else ""),
                        (658, 31),
                        (255, 199, 94),
                        0.68,
                    )
                    label(canvas, f"AI2 / episode {ep} / {name}", (18, 445), size=0.7)
                    label(
                        canvas,
                        f"{n / fps:.2f}s -> {outn / fps:.2f}s   ({n / outn:.2f}x)",
                        (18, 480),
                        size=0.7,
                    )
                    label(
                        canvas,
                        "Same arm paths; different start/wait schedule.",
                        (18, 515),
                        size=0.58,
                    )
                    label(
                        canvas,
                        "Source-pixel edit. NOT collision/rollout validated.",
                        (18, 547),
                        (255, 199, 94),
                        0.56,
                    )
                    label(
                        canvas, "Terminal freeze is display-only.", (18, 579), size=0.53
                    )
                    label(
                        canvas,
                        f"Left source {l / fps:.2f}s",
                        (650, 432),
                        (113, 190, 255),
                        0.51,
                    )
                    label(
                        canvas,
                        f"Right source {r / fps:.2f}s",
                        (970, 432),
                        (255, 170, 103),
                        0.51,
                    )
                    canvas[440:620, 640:960] = cv2.resize(frames[l], (320, 180))
                    canvas[440:620, 960:1280] = cv2.resize(frames[r], (320, 180))
                    if not ok:
                        label(
                            canvas,
                            "MASK MISSING: inspect source views",
                            (660, 395),
                            (255, 90, 90),
                            0.6,
                        )
                    elif overlap > 60:
                        label(
                            canvas,
                            f"PROJECTED ARM OVERLAP: {overlap}px",
                            (658, 395),
                            (255, 105, 90),
                            0.58,
                        )
                    yield canvas

            write_video(target, generate(), fps, 1280, 624)
            if len(tiles) == 12:
                cv2.imwrite(
                    str(d / (name + "_contact.jpg")),
                    cv2.cvtColor(
                        np.vstack(
                            [np.hstack(tiles[i : i + 4]) for i in range(0, 12, 4)]
                        ),
                        cv2.COLOR_RGB2BGR,
                    ),
                )
            check = cv2.VideoCapture(str(target))
            count = 0
            while check.grab():
                count += 1
            check.release()
            if count != total:
                raise RuntimeError("Rendered frame count mismatch")
            row = dict(
                **{
                    k: v
                    for k, v in plan.items()
                    if k
                    not in ("projected_overlap_frames", "physical_collision_verified")
                },
                path=str(target.relative_to(root)),
                video_frames=total,
                video_width=1280,
                video_height=624,
                bytes=target.stat().st_size,
                sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                projected_overlap_frames=int(np.sum(np.array(counts) > 60)),
                max_projected_overlap_pixels=int(max(counts)),
                missing_mask_frames=int(sum(missing)),
                mask_method="SAM3 semantic union, root-geodesic partition",
                render_method="Serial-source scene factorization; opaque arm/object edits plus source shadow deltas",
                physical_collision_verified=False,
            )
            receipts.append(row)
            (
                d
                / (
                    "render_screened_receipts.json"
                    if args.screened
                    else "render_receipts.json"
                )
            ).write_text(json.dumps(receipts, indent=2))
            print(
                "RENDERED",
                ep,
                name,
                row["bytes"],
                row["projected_overlap_frames"],
                row["missing_mask_frames"],
                flush=True,
            )
        masks.cache_clear()
        left_layer.cache_clear()
        del frames, packed
    (
        root
        / ("render_screened_complete.json" if args.screened else "render_complete.json")
    ).write_text(json.dumps(dict(episodes=episodes, complete=True)))


if __name__ == "__main__":
    run_stage(main, renderer=True, segmentation=False)
