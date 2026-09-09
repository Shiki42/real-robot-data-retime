"""Photometric motion evidence, independent of predicted robot masks."""

from dataclasses import dataclass

import cv2
import numpy as np

from ..background.clean_plate import match_background_colors
from .discovery import motion_and_grippers


@dataclass
class PhotometricMotion:
    raw: dict
    discovery: dict
    support: np.ndarray
    ambiguous: np.ndarray
    coefficients: np.ndarray


def opaque_motion_support(frames, background):
    """Positive evidence for dark opaque robot pixels, not a semantic mask.

    Brightening can uncover background. Color-preserving attenuation can be a
    shadow or gray foreground; classify it as ambiguous, never as confirmed
    background. Discovery retains those pixels, but the audit cannot demand
    that a robot mask cover them. This cue cannot certify reflective parts.
    """
    n, h, w = frames.shape[:3]
    if background.shape != (h, w, 3):
        raise ValueError("background and motion frame geometry differ")
    bg = background.astype(np.float32)
    norm2 = np.sum(bg * bg, axis=-1)
    norm = np.sqrt(norm2)
    bg_gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY).astype(np.int16)
    support = np.zeros((n, h, w), bool)
    ambiguous = np.zeros_like(support)
    for t, frame in enumerate(frames):
        pixels = frame.astype(np.float32)
        alpha = np.sum(pixels * bg, axis=-1) / np.maximum(norm2, 1)
        residual = np.linalg.norm(pixels - alpha[..., None] * bg, axis=-1) / np.maximum(
            norm, 1
        )
        attenuation = (alpha >= 0.4) & (alpha < 0.98) & (residual < 0.08)
        darkening = bg_gray - cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.int16)
        dark = frame.max(axis=-1) < 115
        support[t] = dark & (darkening > 25) & ~attenuation
        changed = np.max(cv2.absdiff(frame, background), axis=-1) > 25
        ambiguous[t] = dark & changed & ~support[t]
    return support, ambiguous


def photometric_motion(frames):
    """Normalize independent background evidence; preserve original source RGB."""
    initial = motion_and_grippers(frames)
    normalized, coefficients = match_background_colors(frames, initial["masks"] > 0)
    discovery = motion_and_grippers(normalized)
    support, ambiguous = opaque_motion_support(normalized, discovery["background"])
    discovery["prompt_support"] = support
    return PhotometricMotion(initial, discovery, support, ambiguous, coefficients)
