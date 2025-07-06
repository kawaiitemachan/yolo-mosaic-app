import os
from pathlib import Path

# ディレクトリ設定
BASE_DIR = Path(__file__).parent.parent
PROJECT_ROOT = BASE_DIR  # 互換性のため
DATA_DIR = BASE_DIR / "data"
IMAGES_DIR = DATA_DIR / "images"
ANNOTATIONS_DIR = DATA_DIR / "annotations"
MODELS_DIR = DATA_DIR / "models"
DATASETS_DIR = BASE_DIR / "datasets"

DEFAULT_CONFIG = {
    "annotation": {
        "default_label": "object",
        "labels": ["object", "face", "person", "car", "license_plate"],
        "colors": {
            "object": "#FF0000",      # 赤
            "face": "#00FF00",        # 緑
            "person": "#0000FF",      # 青
            "car": "#FFFF00",         # 黄色
            "license_plate": "#FF00FF" # マゼンタ
        }
    },
    "training": {
        "batch_size": 16,
        "epochs": 100,
        "imgsz": 640,
        "patience": 50,
        "model": "yolo11n-seg.pt",
        "device": "auto"
    },
    "inference": {
        "confidence": 0.25,
        "iou": 0.45,
        "blur_strength": 15,
        "blur_type": "gaussian"
    }
}

def ensure_directories():
    """必要なディレクトリを作成"""
    for dir_path in [DATA_DIR, IMAGES_DIR, ANNOTATIONS_DIR, MODELS_DIR, DATASETS_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)