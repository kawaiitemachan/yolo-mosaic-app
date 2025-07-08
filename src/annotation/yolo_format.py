from pathlib import Path
from PySide6.QtCore import QPointF
from ..config import ANNOTATIONS_DIR, DEFAULT_CONFIG
import shutil

def save_yolo_segmentation(image_path, polygons, img_width, img_height):
    """
    YOLO形式でセグメンテーションアノテーションを保存
    """
    image_path = Path(image_path)
    
    labels = DEFAULT_CONFIG["annotation"]["labels"]
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    
    txt_path = ANNOTATIONS_DIR / f"{image_path.stem}.txt"
    
    lines = []
    for polygon_data in polygons:
        label = polygon_data["label"]
        points = polygon_data["points"]
        
        if label not in label_to_idx:
            continue
        
        class_idx = label_to_idx[label]
        
        normalized_points = []
        for point in points:
            x_norm = point.x() / img_width
            y_norm = point.y() / img_height
            normalized_points.extend([x_norm, y_norm])
        
        line = f"{class_idx} " + " ".join(f"{coord:.6f}" for coord in normalized_points)
        lines.append(line)
    
    with open(txt_path, 'w') as f:
        f.write('\n'.join(lines))
    
    save_classes_file()
    return txt_path

def save_classes_file():
    """
    classes.txtファイルを保存
    """
    labels = DEFAULT_CONFIG["annotation"]["labels"]
    classes_path = ANNOTATIONS_DIR / "classes.txt"
    
    with open(classes_path, 'w') as f:
        f.write('\n'.join(labels))

def create_data_yaml(train_path, val_path, test_path=None):
    """
    YOLOトレーニング用のdata.yamlファイルを作成
    """
    import yaml
    
    labels = DEFAULT_CONFIG["annotation"]["labels"]
    
    data = {
        'path': str(ANNOTATIONS_DIR.parent),
        'train': str(train_path),
        'val': str(val_path),
        'nc': len(labels),
        'names': labels
    }
    
    if test_path:
        data['test'] = str(test_path)
    
    yaml_path = ANNOTATIONS_DIR / "data.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
    
    return yaml_path

def load_yolo_segmentation(txt_path, img_width, img_height, idx_to_label=None):
    """
    YOLO形式のセグメンテーションアノテーションを読み込む
    """
    if idx_to_label is None:
        labels = DEFAULT_CONFIG["annotation"]["labels"]
        idx_to_label = {idx: label for idx, label in enumerate(labels)}
    
    polygons = []
    
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:  # 最低でもクラスIDと3点（6座標）が必要
                continue
                
            class_idx = int(parts[0])
            label = idx_to_label.get(class_idx, "object")
            
            points = []
            for i in range(1, len(parts), 2):
                if i + 1 < len(parts):
                    x = float(parts[i]) * img_width
                    y = float(parts[i + 1]) * img_height
                    points.append(QPointF(x, y))
            
            if len(points) >= 3:
                polygons.append({
                    "points": points,
                    "label": label
                })
    
    return polygons

def save_yolo_segmentation_to_dataset(image_path, polygons, img_width, img_height, dataset_path, split="train"):
    """
    データセット構造に従ってYOLO形式でセグメンテーションアノテーションを保存
    """
    image_path = Path(image_path)
    dataset_path = Path(dataset_path)
    
    # 画像をデータセットにコピー
    dst_image_dir = dataset_path / split / "images"
    dst_image_dir.mkdir(parents=True, exist_ok=True)
    dst_image_path = dst_image_dir / image_path.name
    
    if not dst_image_path.exists():
        shutil.copy2(image_path, dst_image_path)
    
    # ラベルを保存
    dst_label_dir = dataset_path / split / "labels"
    dst_label_dir.mkdir(parents=True, exist_ok=True)
    txt_path = dst_label_dir / f"{image_path.stem}.txt"
    
    # data.yamlからラベル情報を取得
    import yaml
    yaml_path = dataset_path / "data.yaml"
    if yaml_path.exists():
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        labels = data.get('names', DEFAULT_CONFIG["annotation"]["labels"])
    else:
        labels = DEFAULT_CONFIG["annotation"]["labels"]
    
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    
    lines = []
    for polygon_data in polygons:
        label = polygon_data["label"]
        points = polygon_data["points"]
        
        if label not in label_to_idx:
            continue
        
        class_idx = label_to_idx[label]
        
        normalized_points = []
        for point in points:
            x_norm = point.x() / img_width
            y_norm = point.y() / img_height
            normalized_points.extend([x_norm, y_norm])
        
        line = f"{class_idx} " + " ".join(f"{coord:.6f}" for coord in normalized_points)
        lines.append(line)
    
    with open(txt_path, 'w') as f:
        f.write('\n'.join(lines))
    
    return txt_path

def load_yolo_segmentation_from_dataset(image_path, dataset_path, split, img_width, img_height):
    """
    データセット構造からYOLO形式のセグメンテーションアノテーションを読み込む
    """
    image_path = Path(image_path)
    dataset_path = Path(dataset_path)
    
    # ラベルファイルパス
    txt_path = dataset_path / split / "labels" / f"{image_path.stem}.txt"
    
    if not txt_path.exists():
        return []
    
    # data.yamlからラベル情報を取得
    import yaml
    yaml_path = dataset_path / "data.yaml"
    if yaml_path.exists():
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        labels = data.get('names', DEFAULT_CONFIG["annotation"]["labels"])
    else:
        labels = DEFAULT_CONFIG["annotation"]["labels"]
    
    idx_to_label = {idx: label for idx, label in enumerate(labels)}
    
    return load_yolo_segmentation(txt_path, img_width, img_height, idx_to_label)