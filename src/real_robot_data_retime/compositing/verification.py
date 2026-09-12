"""Independent appearance checks on rendered object-origin regions."""

import cv2
import numpy as np


class OriginAudit:
    def __init__(self, frames, objects, episodes):
        self.items = []
        h, w = frames.shape[1:3]
        for event in episodes:
            yy, xx = np.where(objects[event["object_id"], 0])
            if not len(xx):
                raise ValueError("cannot verify an object without its initial mask")
            roi = (
                slice(max(0, yy.min() - 4), min(h, yy.max() + 5)),
                slice(max(0, xx.min() - 4), min(w, xx.max() + 5)),
            )
            template = cv2.cvtColor(frames[0][roi], cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(frames[0], cv2.COLOR_BGR2HSV)
            pixels = hsv[yy, xx]
            saturated = pixels[pixels[:, 1] > 70]
            color = None
            if len(saturated) > len(pixels) * 0.5:
                angles = saturated[:, 0].astype(float) * (2 * np.pi / 180)
                hue = float(
                    np.arctan2(np.sin(angles).mean(), np.cos(angles).mean())
                    * 180
                    / (2 * np.pi)
                    % 180
                )
                saturation = max(50.0, float(np.percentile(saturated[:, 1], 10)) * 0.65)
                color = (hue, saturation)
            if color is not None and self.color_count(frames[0][roi], color) < max(
                3, len(xx) * 0.2
            ):
                color = None
            if color is None and template.std() < 3:
                raise ValueError("object origin has insufficient appearance contrast")
            self.items.append(
                dict(
                    event=event,
                    roi=roi,
                    template=template,
                    origin_mask=objects[event["object_id"], 0][roi].astype(bool),
                    color=color,
                    color_reference=self.color_count(frames[0][roi], color)
                    if color
                    else 0,
                    observations=0,
                    duplicates=0,
                )
            )

    def observe(self, frame, source_frames, times, foreground):
        for item in self.items:
            event = item["event"]
            side = ["left", "right"].index(event["robot_id"])
            t = times[side]
            if t <= event["grasp_frame"] + 5:
                continue
            roi = item["roi"]
            hidden = np.asarray(foreground[roi], dtype=bool)
            if hidden.mean() > 0.1 or hidden[item["origin_mask"]].mean() > 0.1:
                continue
            visible = ~hidden
            if item["color"] is not None:
                expected = self.color_count(
                    source_frames[t][roi], item["color"], visible
                )
                rendered = self.color_count(frame[roi], item["color"], visible)
                absent = expected < max(2, item["color_reference"] * 0.1)
                duplicate = rendered - expected > max(3, item["color_reference"] * 0.05)
            else:
                expected = cv2.cvtColor(source_frames[t][roi], cv2.COLOR_BGR2GRAY)
                rendered = cv2.cvtColor(frame[roi], cv2.COLOR_BGR2GRAY)
                reference = item["template"][visible].astype(float)
                if reference.std() < 3:
                    continue
                reference -= reference.mean()
                scores = []
                for image in (expected, rendered):
                    values = image[visible].astype(float)
                    values -= values.mean()
                    norm = np.linalg.norm(values) * np.linalg.norm(reference)
                    scores.append(
                        float(values @ reference / norm) if norm > 1e-8 else 0.0
                    )
                source_score, rendered_score = scores
                absent = source_score < 0.5
                duplicate = rendered_score > 0.8
            if absent:
                item["observations"] += 1
                item["duplicates"] += int(duplicate)

    @staticmethod
    def color_count(image, color, visible=None):
        hue, saturation = color
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(float)
        delta = np.abs(hsv[:, :, 0] - hue)
        match = (
            (np.minimum(delta, 180 - delta) < 10)
            & (hsv[:, :, 1] > saturation)
            & (hsv[:, :, 2] > 25)
        )
        if visible is not None:
            match &= visible
        return int(match.sum())

    def report(self):
        entries = [
            dict(
                object_id=i["event"]["object_id"],
                appearance_method="chromatic_occupancy"
                if i["color"]
                else "template_correlation",
                robot_id=i["event"]["robot_id"],
                clear_origin_observations=i["observations"],
                duplicate_observations=i["duplicates"],
                passed=i["observations"] >= 3 and i["duplicates"] == 0,
            )
            for i in self.items
        ]
        return dict(
            passed=all(e["passed"] for e in entries),
            objects=entries,
            method="rendered_origin_appearance_against_observed_source_absence",
            occlusion_policy="per_pixel_foreground_exclusion",
        )
