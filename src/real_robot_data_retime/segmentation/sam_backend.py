class SamVideo:
    def __init__(self, model_id="facebook/sam2.1-hiera-small", device="cuda"):
        import torch
        from transformers import Sam2VideoModel, Sam2VideoProcessor

        self.model_id = model_id
        self.torch = torch
        self.device = device
        self.processor = Sam2VideoProcessor.from_pretrained(model_id)
        self.model = Sam2VideoModel.from_pretrained(model_id).to(device)

    def propagate(self, frames, proposals, seed_frame=0, reverse=False):
        """Stream automatically prompted masks with bounded frame/memory cache.

        Reverse uses a separate session seeded at the same automatically chosen
        frame. Visibility and geometry validation remain the caller's job.
        """
        import cv2
        from PIL import Image
        import numpy as np

        session = self.processor.init_video_session(
            inference_device=self.device,
            processing_device="cpu",
            video_storage_device="cpu",
            dtype=self.torch.float32,
        )
        ids = list(range(1, len(proposals) + 1))
        boxes = []
        for p in proposals:
            x, y, w, h = map(float, p["bbox"])
            boxes.append([x, y, x + w, y + h])
        if not boxes:
            raise ValueError("no automatic segmentation proposals")
        prompts = {}
        if all("mask" in p for p in proposals):
            centers = []
            for p in proposals:
                distance = cv2.distanceTransform(
                    p["mask"].astype(np.uint8), cv2.DIST_L2, 5
                )
                y, x = np.unravel_index(np.argmax(distance), distance.shape)
                centers.append([float(x), float(y)])
            points = []
            labels = []
            for k, c in enumerate(centers):
                others = [z for j, z in enumerate(centers) if j != k]
                positives = proposals[k].get("positive_points", [c])
                points.append([*positives, *others])
                labels.append([*([1] * len(positives)), *([0] * len(others))])
            prompts = dict(input_points=[points], input_labels=[labels])
        self.processor.add_inputs_to_inference_session(
            session,
            frame_idx=0,
            obj_ids=ids,
            input_boxes=[boxes],
            original_size=frames.shape[1:3],
            **prompts,
        )
        indices = (
            range(seed_frame, -1, -1) if reverse else range(seed_frame, len(frames))
        )
        with self.torch.inference_mode():
            for local, t in enumerate(indices):
                image = Image.fromarray(cv2.cvtColor(frames[t], cv2.COLOR_BGR2RGB))
                values = self.processor(images=image, return_tensors="pt")[
                    "pixel_values"
                ]
                output = self.model(session, frame=values, frame_idx=local)
                masks = self.processor.post_process_masks(
                    [output.pred_masks],
                    original_sizes=[frames.shape[1:3]],
                    binarize=True,
                )[0]
                yield t, masks[:, 0].cpu().numpy()
                # Features/memory, not the full processed video, drive propagation.
                for key in list(session.processed_frames):
                    if key < local - 1:
                        del session.processed_frames[key]
                for obj in session.output_dict_per_obj.values():
                    history = obj["non_cond_frame_outputs"]
                    for key in list(history):
                        if key < local - 32:
                            del history[key]
