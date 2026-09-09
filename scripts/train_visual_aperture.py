"""Offline visual supervision from recorded sensors; inference needs only RGB.

Episode IDs 0/1 are reserved for final testing; every seventh remaining
episode is used for validation. No labels
or timings from those episodes enter optimization or per-episode configuration.
"""

import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from real_robot_data_retime.interaction.visual_aperture import (
    build_model,
    prepare_images,
)
from torchvision.transforms import v2


class ApertureImages(Dataset):
    def __init__(self, root, split, task):
        self.images = []
        self.labels = []
        self.indices = []
        self.episodes = []
        self.motion = []
        self.weights = []
        for path in sorted(root.glob(f"{task}_*.npz")):
            episode = int(path.stem.rsplit("_", 1)[1])
            if episode < 2:
                continue
            validation = episode % 7 == 0
            if validation != (split == "validation"):
                continue
            image_path = path.with_suffix(".npy")
            if not image_path.exists():
                np.save(image_path, np.load(path)["frames"])
            images = np.load(image_path, mmap_mode="r")
            labels = np.load(path)["aperture"] / 70.0
            offset = len(self.images)
            self.images.append(images)
            self.labels.append(labels)
            delta = np.max(np.abs(np.diff(labels, axis=0, prepend=labels[:1])), axis=1)
            moving = (
                np.convolve((delta > 0.025).astype(float), np.ones(7), mode="same") > 0
            )
            self.motion.append(moving)
            self.weights.extend(np.where(moving, 5.0, 1.0).tolist())
            self.indices.extend((offset, i) for i in range(len(images)))
            self.episodes.append(episode)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, k):
        episode, frame = self.indices[k]
        image = torch.from_numpy(
            np.array(self.images[episode][frame], copy=True)
        ).permute(2, 0, 1)
        return (
            image,
            torch.from_numpy(self.labels[episode][frame]),
            self.motion[episode][frame],
        )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--epochs", type=int, default=16)
    p.add_argument("--initial-checkpoint", type=Path)
    a = p.parse_args()
    torch.manual_seed(42)
    np.random.seed(42)
    torch.set_num_threads(4)
    train = ApertureImages(a.data, "train", a.task)
    val = ApertureImages(a.data, "validation", a.task)
    if not len(train) or not len(val):
        raise ValueError("empty episode split")
    a.output.mkdir(parents=True, exist_ok=True)
    print("splits", train.episodes, val.episodes, len(train), len(val), flush=True)
    loaders = [
        DataLoader(
            d,
            batch_size=32,
            sampler=torch.utils.data.WeightedRandomSampler(
                d.weights, len(d), replacement=True
            )
            if i == 0
            else None,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
        )
        for i, d in enumerate([train, val])
    ]
    model = build_model().cuda()
    if a.initial_checkpoint:
        payload = torch.load(
            a.initial_checkpoint, map_location="cpu", weights_only=True
        )
        model.load_state_dict(payload["model"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    normalize = v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    augment = v2.Compose(
        [
            v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            v2.RandomAffine(degrees=12, translate=(0.18, 0.12), scale=(0.85, 1.15)),
        ]
    )
    best = float("inf")
    history = []
    for epoch in range(a.epochs):
        model.train()
        losses = []
        for x, y, moving in loaders[0]:
            x = x.cuda(non_blocking=True).float() / 255
            y = y.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                # Automatically randomize low-saturation bright surfaces to
                # expose dark-table appearances without changing jaw geometry.
                brightness = x.amax(dim=1, keepdim=True)
                saturation = (
                    brightness - x.amin(dim=1, keepdim=True)
                ) / brightness.clamp_min(0.01)
                surface = (brightness > 0.42) & (saturation < 0.28)
                tint = torch.rand((len(x), 3, 1, 1), device=x.device) * 0.8 + 0.05
                blend = torch.rand((len(x), 1, 1, 1), device=x.device)
                jittered = torch.where(surface, x * (1 - blend) + tint * blend, x)
                pred = model(normalize(prepare_images(augment(jittered))))
                loss = nn.functional.smooth_l1_loss(pred, y, beta=0.1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
        model.eval()
        targets = []
        preds = []
        movements = []
        with torch.inference_mode():
            for x, y, moving in loaders[1]:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = model(normalize(prepare_images(x.cuda().float() / 255)))
                preds.append(out.float().cpu().numpy() * 70)
                targets.append(y.numpy() * 70)
                movements.append(moving.numpy())
        predictions = np.concatenate(preds)
        targets = np.concatenate(targets)
        mae = np.mean(np.abs(predictions - targets), axis=0)
        motion_mae = np.mean(
            np.abs(predictions - targets)[np.concatenate(movements)], axis=0
        )
        p90 = np.percentile(np.abs(predictions - targets), 90, axis=0)
        row = dict(
            epoch=epoch,
            train_loss=float(np.mean(losses)),
            validation_mae_mm=mae.tolist(),
            validation_p90_mm=p90.tolist(),
            validation_motion_mae_mm=motion_mae.tolist(),
        )
        history.append(row)
        print(row, flush=True)
        metric = 0.5 * mae.mean() + 0.5 * motion_mae.mean()
        if metric < best:
            best = float(metric)
            torch.save(
                {
                    "model": model.state_dict(),
                    "preprocessing": "local_contrast_stride16_v1",
                    "task": a.task,
                    "train_episodes": train.episodes,
                    "validation_episodes": val.episodes,
                    "metrics": row,
                },
                a.output / "best.pt",
            )
            np.savez_compressed(
                a.output / "validation_predictions.npz",
                predictions=predictions,
                targets=targets,
            )
        (a.output / "history.json").write_text(json.dumps(history, indent=2))


if __name__ == "__main__":
    main()
