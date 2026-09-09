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
            if template.std() < 3:
                raise ValueError("object origin has insufficient appearance contrast")
            self.items.append(
                dict(
                    event=event,
                    roi=roi,
                    template=template,
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
            if foreground[roi].mean() > 0.1:
                continue
            expected = cv2.cvtColor(source_frames[t][roi], cv2.COLOR_BGR2GRAY)
            rendered = cv2.cvtColor(frame[roi], cv2.COLOR_BGR2GRAY)
            source_score = float(
                cv2.matchTemplate(expected, item["template"], cv2.TM_CCOEFF_NORMED)[
                    0, 0
                ]
            )
            rendered_score = float(
                cv2.matchTemplate(rendered, item["template"], cv2.TM_CCOEFF_NORMED)[
                    0, 0
                ]
            )
            if source_score < 0.5:
                item["observations"] += 1
                item["duplicates"] += int(rendered_score > 0.8)

    def report(self):
        entries = [
            dict(
                object_id=i["event"]["object_id"],
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
        )
