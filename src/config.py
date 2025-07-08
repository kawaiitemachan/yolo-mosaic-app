import os
from pathlib import Path
from .utils.app_paths import get_data_path, get_datasets_path, get_models_path, get_application_path

# ディレクトリ設定
BASE_DIR = get_application_path()
PROJECT_ROOT = BASE_DIR  # 互換性のため
DATA_DIR = get_data_path()
IMAGES_DIR = get_data_path("images")
ANNOTATIONS_DIR = get_data_path("annotations")
MODELS_DIR = get_models_path()
DATASETS_DIR = get_datasets_path()

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