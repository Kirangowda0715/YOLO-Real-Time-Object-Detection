"""
Modern YOLO Object Detection & Tracking Module
Supports YOLO11, YOLOv8 (via Ultralytics with GPU CUDA acceleration)
as well as legacy Darknet YOLOv3 / YOLOv3-tiny (via OpenCV DNN).
"""
import os
import time
import cv2
import numpy as np


def generate_palette(num_classes=80):
    np.random.seed(42)
    colors = np.random.randint(40, 240, size=(num_classes, 3), dtype=np.uint8)
    return [tuple(map(int, c)) for c in colors]


class YOLODetector:
    def __init__(
        self,
        model="yolo11n",
        conf_threshold=0.35,
        iou_threshold=0.45,
        device=None,
    ):
        self.model_name = model
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        # Determine backend engine: 'ultralytics' or 'opencv'
        is_legacy = model in ("tiny", "full", "yolov3", "yolov3-tiny") or (
            isinstance(model, str) and (model.endswith(".weights") or model.endswith(".cfg"))
        )

        if is_legacy:
            self.engine = "opencv"
            self._init_opencv(model)
        else:
            self.engine = "ultralytics"
            self._init_ultralytics(model, device)

    def _init_ultralytics(self, model, device):
        import torch
        from ultralytics import YOLO

        if device is None:
            self.device = 0 if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.device_name = (
            torch.cuda.get_device_name(self.device)
            if (isinstance(self.device, int) or str(self.device).isdigit()) and torch.cuda.is_available()
            else "CPU"
        )

        # Append .pt if not provided
        model_file = f"{model}.pt" if not model.endswith(".pt") else model
        print(f"Loading modern YOLO model ({model_file}) on {self.device_name}...")
        self.model = YOLO(model_file)
        self.classes = list(self.model.names.values())
        self.colors = generate_palette(max(80, len(self.classes)))
        print(f"Model ready ({len(self.classes)} classes) with GPU acceleration.")

    def _init_opencv(self, model):
        if model in ("tiny", "yolov3-tiny"):
            self.weights_path = os.path.join("weights", "yolov3-tiny.weights")
            self.cfg_path = os.path.join("cfg", "yolov3-tiny.cfg")
        else:
            self.weights_path = os.path.join("weights", "yolov3.weights")
            self.cfg_path = os.path.join("cfg", "yolov3.cfg")

        with open("coco.names", "r", encoding="utf-8") as f:
            self.classes = [line.strip() for line in f.readlines() if line.strip()]

        self.colors = generate_palette(len(self.classes))
        self.device_name = "CPU (OpenCV DNN)"
        print(f"Loading legacy YOLO model from {self.weights_path}...")
        self.net = cv2.dnn.readNet(self.weights_path, self.cfg_path)
        self.output_layers = self.net.getUnconnectedOutLayersNames()
        print("Legacy model ready.")

    def reset_tracker(self):
        """
        Resets ByteTrack tracking state and clears ID counter so tracks start from #1 again.
        """
        if self.engine == "ultralytics":
            try:
                from ultralytics.trackers.basetrack import BaseTrack
                BaseTrack._count = 0
            except Exception:
                pass
            try:
                if hasattr(self.model, "predictor") and self.model.predictor is not None:
                    if hasattr(self.model.predictor, "trackers"):
                        self.model.predictor.trackers = []
            except Exception:
                pass

    def detect(self, frame, track=True):
        """
        Runs object detection (and tracking if available) on frame.
        Returns:
            list of dict: [{ 'box': [x, y, w, h], 'class_id': int, 'label': str,
                             'confidence': float, 'color': (b,g,r), 'track_id': int|None }]
            float: inference time in milliseconds
        """
        if self.engine == "ultralytics":
            return self._detect_ultralytics(frame, track=track)
        else:
            return self._detect_opencv(frame)

    def _detect_ultralytics(self, frame, track=True):
        t_start = time.perf_counter()
        if track:
            results = self.model.track(
                frame,
                persist=True,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                device=self.device,
                verbose=False,
            )
        else:
            results = self.model.predict(
                frame,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                device=self.device,
                verbose=False,
            )
        inference_time_ms = (time.perf_counter() - t_start) * 1000

        detections = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            xyxy = boxes.xyxy.cpu().numpy()
            conf = boxes.conf.cpu().numpy()
            cls_ids = boxes.cls.cpu().numpy().astype(int)
            track_ids = (
                boxes.id.int().cpu().numpy()
                if boxes.id is not None
                else [None] * len(xyxy)
            )

            for i in range(len(xyxy)):
                x1, y1, x2, y2 = xyxy[i]
                w = int(x2 - x1)
                h = int(y2 - y1)
                cid = int(cls_ids[i])
                tid = int(track_ids[i]) if track_ids[i] is not None else None
                label_name = self.classes[cid] if cid < len(self.classes) else f"cls_{cid}"

                detections.append(
                    {
                        "box": [max(0, int(x1)), max(0, int(y1)), w, h],
                        "class_id": cid,
                        "label": label_name,
                        "track_id": tid,
                        "confidence": float(conf[i]),
                        "color": self.colors[cid % len(self.colors)],
                    }
                )

        return detections, inference_time_ms

    def _detect_opencv(self, frame):
        height, width = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        t_start = time.perf_counter()
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)
        inference_time_ms = (time.perf_counter() - t_start) * 1000

        boxes = []
        confidences = []
        class_ids = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                cid = int(np.argmax(scores))
                conf = float(scores[cid])
                if conf > self.conf_threshold:
                    cx = int(detection[0] * width)
                    cy = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(cx - w / 2)
                    y = int(cy - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(conf)
                    class_ids.append(cid)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.iou_threshold)
        detections = []
        if len(indexes) > 0:
            flat = indexes.flatten() if hasattr(indexes, "flatten") else [i[0] if isinstance(i, (list, tuple, np.ndarray)) else i for i in indexes]
            for i in flat:
                x, y, w, h = boxes[i]
                cid = class_ids[i]
                detections.append(
                    {
                        "box": [max(0, x), max(0, y), w, h],
                        "class_id": cid,
                        "label": self.classes[cid] if cid < len(self.classes) else f"cls_{cid}",
                        "track_id": None,
                        "confidence": confidences[i],
                        "color": self.colors[cid % len(self.colors)],
                    }
                )

        return detections, inference_time_ms

    def draw_detections(self, frame, detections, fps=None, inference_time_ms=None, show_hud=True):
        """
        Draws modern, high-contrast bounding boxes with tracking IDs and optional top status HUD.
        """
        annotated = frame.copy()
        height, width = annotated.shape[:2]

        scale_factor = max(0.5, min(2.5, width / 1280.0))
        box_thickness = max(2, int(scale_factor * 2))
        font_scale = 0.5 * scale_factor
        text_thickness = max(1, int(scale_factor * 1.5))
        hud_height = max(42, int(42 * scale_factor))

        for d in detections:
            x, y, w, h = d["box"]
            color = d["color"]
            tid = f" #{d['track_id']}" if d.get("track_id") is not None else ""
            label = f"{d['label']}{tid} {int(d['confidence'] * 100)}%"

            # Bounding box
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, box_thickness)

            # Badge
            font = cv2.FONT_HERSHEY_SIMPLEX
            (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, text_thickness)
            pad = int(4 * scale_factor)
            badge_y1 = max(0, y - text_h - pad * 2)
            badge_y2 = y
            badge_x1 = x
            badge_x2 = x + text_w + pad * 2

            cv2.rectangle(annotated, (badge_x1, badge_y1), (badge_x2, badge_y2), color, -1)
            text_color = (0, 0, 0) if (color[0] * 0.114 + color[1] * 0.587 + color[2] * 0.299) > 160 else (255, 255, 255)
            cv2.putText(
                annotated,
                label,
                (badge_x1 + pad, badge_y2 - pad),
                font,
                font_scale,
                text_color,
                text_thickness,
                cv2.LINE_AA,
            )

        if not show_hud:
            return annotated

        # Top HUD bar
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, 0), (width, hud_height), (20, 20, 25), -1)
        cv2.addWeighted(overlay, 0.75, annotated, 0.25, 0, annotated)

        stats_parts = []
        if fps is not None:
            stats_parts.append(f"FPS: {fps:.1f}")
        if inference_time_ms is not None:
            stats_parts.append(f"Latency: {inference_time_ms:.1f}ms")
        stats_parts.append(f"Objects: {len(detections)}")
        stats_parts.append(f"Device: {self.device_name}")

        hud_text = "  |  ".join(stats_parts)
        cv2.putText(
            annotated,
            hud_text,
            (int(12 * scale_factor), int(hud_height * 0.65)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55 * scale_factor,
            (0, 255, 180),
            text_thickness,
            cv2.LINE_AA,
        )

        hint = "[P] Pause  [S] Save  [ESC] Exit"
        (hint_w, _), _ = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.45 * scale_factor, text_thickness)
        if width > hint_w + int(450 * scale_factor):
            cv2.putText(
                annotated,
                hint,
                (width - hint_w - int(12 * scale_factor), int(hud_height * 0.65)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45 * scale_factor,
                (200, 200, 200),
                text_thickness,
                cv2.LINE_AA,
            )

        return annotated
