# -*- coding: utf-8 -*-
"""
四模型管理器（单例）- 全部基于 ultralytics 框架
- 对比模型: YOLOv8s / YOLOv8n (ultralytics) × YOLOv11s / YOLOv11n (ultralytics)
  * s 系列: 精度优先，图片检测主力
  * n 系列: 轻量高速，视频/摄像头实时检测主力
- 训推分离: 模型权重仅在进程内加载一次，之后所有请求复用（线程锁保证并发安全）
- 权重优先级: webapp/models/ 下的 Kaggle 训练权重 > COCO 预训练演示权重
  * yolov8s_traffic_best.pt 存在 -> YOLOv8s 使用交通数据集训练权重(11类)
  * 否则回退到 COCO 预训练演示模式, 通过类别映射输出交通类别
"""
import json
import os
import threading
from pathlib import Path

import numpy as np
import torch
from PIL import Image

INPUT_SIZE = 320  # 与计划书§5.1 公平性约束一致: 全模型统一输入尺寸 320×320

# 计划书数据集 11 类（Traffic-Object, Kaggle: tailength/traffic-object）
DEFAULT_CLASSES = [
    "Vehicle", "Bus", "Bicycle", "Person", "Engine", "Truck", "Tricycle",
    "Obstacle", "Pothole", "Traffic Light", "Traffic Sign",
]

# COCO 预训练演示模式下的类别映射 -> 交通类别
COCO2TRAFFIC = {
    "car": "Vehicle",
    "bus": "Bus",
    "truck": "Truck",
    "bicycle": "Bicycle",
    "person": "Person",
    "motorcycle": "Engine",
    "traffic light": "Traffic Light",
    "stop sign": "Traffic Sign",
}

# 四模型注册表: 全部使用 ultralytics 框架，API 一致
MODEL_SPECS = {
    "yolov8": {
        "label": "YOLOv8s", "family": "ultralytics",
        "traffic_weight": "yolov8s_traffic_best.pt", "coco_weight": "yolov8s_coco.pt",
    },
    "yolov8n": {
        "label": "YOLOv8n", "family": "ultralytics",
        "traffic_weight": "yolov8n_traffic_best.pt", "coco_weight": "yolov8n_coco.pt",
    },
    "yolov11": {
        "label": "YOLOv11s", "family": "ultralytics",
        "traffic_weight": "yolov11s_traffic_best.pt", "coco_weight": "yolov11s_coco.pt",
    },
    "yolov11n": {
        "label": "YOLOv11n", "family": "ultralytics",
        "traffic_weight": "yolov11n_traffic_best.pt", "coco_weight": "yolov11n_coco.pt",
    },
}


def project_paths():
    webapp_dir = Path(__file__).resolve().parent.parent.parent  # webapp/
    return webapp_dir, webapp_dir / "models"


class ModelManager:
    _instance = None
    _instance_lock = threading.Lock()

    def __init__(self):
        self.webapp_dir, self.weights_dir = project_paths()
        from django.conf import settings

        self.device = torch.device(
            "cuda" if (getattr(settings, "ML_DEVICE", "auto") == "auto"
                       and torch.cuda.is_available())
            else "cpu")
        self.device_str = "cuda" if self.device.type == "cuda" else "cpu"
        # 类别: 优先读取 Kaggle 导出的 traffic_classes.json
        classes_file = self.weights_dir / "traffic_classes.json"
        self.classes = (json.loads(classes_file.read_text(encoding="utf-8"))
                        if classes_file.exists() else list(DEFAULT_CLASSES))
        self._infer_lock = threading.Lock()
        self.models = {}  # name -> model 对象
        self.modes = {}   # name -> traffic / coco / off
        for name, spec in MODEL_SPECS.items():
            self._load_ultralytics(name, spec)

    # ---------- 单例 ----------
    @classmethod
    def get(cls):
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    # ---------- 模型加载 ----------
    def _load_ultralytics(self, name, spec):
        """加载 ultralytics 系模型(YOLOv8s/YOLOv8n/YOLOv11s/YOLOv11n):
        优先交通训练权重, 否则 COCO 演示"""
        os.environ.setdefault("YOLO_VERBOSE", "False")
        os.environ.setdefault("YOLO_AUTOINSTALL", "False")
        custom = self.weights_dir / spec["traffic_weight"]
        coco = self.weights_dir / spec["coco_weight"]
        self.modes[name] = "off"
        weights_path = custom if custom.exists() else (coco if coco.exists() else None)
        if weights_path is None:
            print(f"[ModelManager] 未找到 {spec['label']} 权重({spec['traffic_weight']} / {spec['coco_weight']})，该模型不可用")
            return
        try:
            import logging

            from ultralytics import YOLO

            logging.getLogger("ultralytics").setLevel(logging.ERROR)
            model = YOLO(str(weights_path))
            model.to(self.device)
            model.fuse()  # 融合 Conv+BN, 加速推理(仅影响推理, 不改权重)
            self.models[name] = model
            self.modes[name] = "traffic" if custom.exists() else "coco"
            print(f"[ModelManager] {spec['label']} 加载完成 mode={self.modes[name]} weights={weights_path.name}")
        except Exception as e:  # 加载失败不应阻断整个系统(其余模型仍可用)
            self.modes[name] = "off"
            print(f"[ModelManager] {spec['label']} 加载失败，已停用: {e}")

    @property
    def available_models(self):
        return [name for name in MODEL_SPECS if self.models.get(name) is not None]

    # 兼容旧代码的属性访问
    @property
    def yolov8_mode(self):
        return self.modes.get("yolov8", "off")

    @property
    def yolov11_mode(self):
        return self.modes.get("yolov11", "off")

    # ---------- 推理 ----------
    def detect(self, model_name, pil_image, conf_threshold):
        """返回统一结构的检测结果列表:
        [{"class": str, "confidence": float, "bbox": [x1, y1, x2, y2]}]  坐标为原图像素坐标
        """
        spec = MODEL_SPECS.get(model_name)
        if spec is None:
            raise ValueError(f"未知模型: {model_name}")
        if self.models.get(model_name) is None:
            raise RuntimeError(f"{spec['label']} 模型不可用：请将权重放入 webapp/models/ 后重启服务")
        return self._detect(model_name, pil_image, conf_threshold)

    def _detect(self, name, pil_image, conf_threshold):
        """ultralytics 统一推理方法（YOLOv8/YOLOv11 共用）"""
        model = self.models[name]
        with self._infer_lock:
            img = pil_image.convert("RGB")
            results = model.predict(
                source=img,
                imgsz=INPUT_SIZE,
                conf=conf_threshold,
                iou=0.45,  # NMS IoU 阈值
                device=self.device_str,
                verbose=False,
            )
            boxes = results[0].boxes
            names = model.names or {}
            detections = []
            if boxes is not None and len(boxes):
                xyxy = boxes.xyxy.cpu().tolist()
                confs = boxes.conf.cpu().tolist()
                clses = boxes.cls.cpu().tolist()
                for (x1, y1, x2, y2), score, cls_id in zip(xyxy, confs, clses):
                    cls_name = names.get(int(cls_id), str(int(cls_id)))
                    if self.modes[name] == "coco":
                        cls_name = COCO2TRAFFIC.get(str(cls_name).lower())
                        if cls_name is None:
                            continue
                    detections.append({
                        "class": cls_name,
                        "confidence": round(float(score), 4),
                        "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                    })
            return detections

    # ---------- 视频帧推理（复用图片推理管道，供视频/摄像头检测调用） ----------
    def detect_video_frame(self, model_name, pil_image, conf_threshold):
        """视频单帧推理：与 detect() 同逻辑，但做了额外的异常兜底，避免单帧失败中断整个视频"""
        try:
            return self.detect(model_name, pil_image, conf_threshold)
        except Exception as e:
            print(f"[ModelManager] 视频帧推理失败(model={model_name}): {e}")
            return []
