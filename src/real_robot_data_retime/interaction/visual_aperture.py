"""Image-only jaw-aperture inference, trained offline on recorded sensors."""

from pathlib import Path
import cv2
import numpy as np
import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights


def build_model(pretrained=True):
    model = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
    model.avgpool = nn.AdaptiveAvgPool2d((2, 3))
    model.fc = nn.Sequential(
        nn.Flatten(),
        nn.Linear(512 * 6, 256),
        nn.ReLU(),
        nn.Dropout(0.15),
        nn.Linear(256, 2),
    )
    return model


def prepare_images(inputs):
    """Local contrast removes table color and slowly varying illumination."""
    smooth = nn.functional.avg_pool2d(inputs, 21, stride=1, padding=10)
    return (inputs - smooth).clamp(-0.5, 0.5) + 0.5


def predict_aperture(frames, checkpoint: Path, batch_size=64):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload["preprocessing"] != "local_contrast_v1":
        raise ValueError("checkpoint preprocessing does not match visual model")
    model = build_model(pretrained=False).cuda().eval()
    model.load_state_dict(payload["model"])
    mean = torch.tensor([0.485, 0.456, 0.406], device="cuda")[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device="cuda")[None, :, None, None]
    result = []
    with torch.inference_mode():
        for start in range(0, len(frames), batch_size):
            images = np.array(
                [
                    cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), (320, 240))
                    for f in frames[start : start + batch_size]
                ]
            )
            inputs = torch.from_numpy(images).permute(0, 3, 1, 2).cuda().float() / 255
            with torch.autocast("cuda", dtype=torch.bfloat16):
                predictions = model((prepare_images(inputs) - mean) / std)
            result.append(predictions.float().cpu().numpy() * 70.0)
    return np.concatenate(result), payload["metrics"]
