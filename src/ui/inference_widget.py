from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                               QLabel, QSlider, QComboBox, QListWidget,
                               QSplitter, QFileDialog, QListWidgetItem,
                               QGroupBox, QDoubleSpinBox, QCheckBox, QSpinBox)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QImage, QPixmap, QPainter
import cv2
import numpy as np
from pathlib import Path

from ..config import MODELS_DIR, DEFAULT_CONFIG
from ..utils.device_utils import get_device

class InferenceThread(QThread):
    result_ready = Signal(np.ndarray, list)
    progress = Signal(str)
    error = Signal(str)
    
    def __init__(self, model_path, image_path, confidence, iou):
        super().__init__()
        self.model_path = model_path
        self.image_path = image_path
        self.confidence = confidence
        self.iou = iou
        
    def run(self):
        try:
            from ultralytics import YOLO
            
            self.progress.emit("モデルを読み込み中...")
            model = YOLO(self.model_path)
            
            device, _ = get_device()
            
            self.progress.emit("推論を実行中...")
            results = model(
                self.image_path,
                conf=self.confidence,
                iou=self.iou,
                device=device
            )
            
            image = cv2.imread(str(self.image_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            detections = []
            for r in results:
                if r.masks is not None:
                    for i, mask in enumerate(r.masks.data):
                        mask_np = mask.cpu().numpy()
                        mask_resized = cv2.resize(mask_np, (image.shape[1], image.shape[0]))
                        
                        cls = int(r.boxes.cls[i])
                        conf = float(r.boxes.conf[i])
                        label = model.names[cls]
                        
                        detections.append({
                            'mask': mask_resized > 0.5,
                            'label': label,
                            'confidence': conf
                        })
            
            self.result_ready.emit(image, detections)
            
        except Exception as e:
            self.error.emit(f"エラー: {str(e)}")

class ImageViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.original_image = None
        self.processed_image = None
        self.pixmap = None
        self.setMinimumSize(400, 300)
        
    def set_image(self, image):
        if isinstance(image, np.ndarray):
            height, width, channel = image.shape
            bytes_per_line = 3 * width
            q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
            self.pixmap = QPixmap.fromImage(q_image)
        else:
            self.pixmap = QPixmap(image)
        self.update()
    
    def paintEvent(self, event):
        if self.pixmap:
            painter = QPainter(self)
            scaled_pixmap = self.pixmap.scaled(
                self.size(), 
                Qt.AspectRatioMode.KeepAspectRatio, 
                Qt.TransformationMode.SmoothTransformation
            )
            x = (self.width() - scaled_pixmap.width()) // 2
            y = (self.height() - scaled_pixmap.height()) // 2
            painter.drawPixmap(x, y, scaled_pixmap)

class InferenceWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.current_image = None
        self.current_image_path = None
        self.current_detections = []
        
    def init_ui(self):
        layout = QHBoxLayout(self)
        
        left_panel = self.create_left_panel()
        
        self.image_viewer = ImageViewer()
        
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(self.image_viewer)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        
        layout.addWidget(splitter)
        
    def create_left_panel(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        model_group = self.create_model_group()
        layout.addWidget(model_group)
        
        param_group = self.create_param_group()
        layout.addWidget(param_group)
        
        mosaic_group = self.create_mosaic_group()
        layout.addWidget(mosaic_group)
        
        load_image_btn = QPushButton("画像を選択")
        load_image_btn.clicked.connect(self.load_image)
        layout.addWidget(load_image_btn)
        
        load_folder_btn = QPushButton("フォルダを選択")
        load_folder_btn.clicked.connect(self.load_folder)
        layout.addWidget(load_folder_btn)
        
        self.file_list = QListWidget()
        self.file_list.itemClicked.connect(self.on_file_selected)
        layout.addWidget(self.file_list)
        
        self.inference_btn = QPushButton("推論実行")
        self.inference_btn.clicked.connect(self.run_inference)
        layout.addWidget(self.inference_btn)
        
        self.apply_mosaic_btn = QPushButton("モザイク適用")
        self.apply_mosaic_btn.clicked.connect(self.apply_mosaic)
        self.apply_mosaic_btn.setEnabled(False)
        layout.addWidget(self.apply_mosaic_btn)
        
        self.save_btn = QPushButton("画像を保存")
        self.save_btn.clicked.connect(self.save_image)
        self.save_btn.setEnabled(False)
        layout.addWidget(self.save_btn)
        
        self.batch_process_btn = QPushButton("バッチ処理")
        self.batch_process_btn.clicked.connect(self.batch_process)
        layout.addWidget(self.batch_process_btn)
        
        layout.addStretch()
        return widget
    
    def create_model_group(self):
        group = QGroupBox("モデル設定")
        layout = QVBoxLayout(group)
        
        self.model_combo = QComboBox()
        self.refresh_models()
        layout.addWidget(self.model_combo)
        
        refresh_btn = QPushButton("モデルリストを更新")
        refresh_btn.clicked.connect(self.refresh_models)
        layout.addWidget(refresh_btn)
        
        return group
    
    def create_param_group(self):
        group = QGroupBox("推論パラメータ")
        layout = QVBoxLayout(group)
        
        conf_layout = QHBoxLayout()
        conf_layout.addWidget(QLabel("信頼度閾値:"))
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.0, 1.0)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setValue(DEFAULT_CONFIG["inference"]["confidence"])
        conf_layout.addWidget(self.conf_spin)
        layout.addLayout(conf_layout)
        
        iou_layout = QHBoxLayout()
        iou_layout.addWidget(QLabel("IoU閾値:"))
        self.iou_spin = QDoubleSpinBox()
        self.iou_spin.setRange(0.0, 1.0)
        self.iou_spin.setSingleStep(0.05)
        self.iou_spin.setValue(DEFAULT_CONFIG["inference"]["iou"])
        iou_layout.addWidget(self.iou_spin)
        layout.addLayout(iou_layout)
        
        return group
    
    def create_mosaic_group(self):
        group = QGroupBox("モザイク設定")
        layout = QVBoxLayout(group)
        
        blur_type_layout = QHBoxLayout()
        blur_type_layout.addWidget(QLabel("モザイクタイプ:"))
        self.blur_type_combo = QComboBox()
        self.blur_type_combo.addItems(["gaussian", "pixelate", "blur", "black", "white", "tile"])
        blur_type_layout.addWidget(self.blur_type_combo)
        layout.addLayout(blur_type_layout)
        
        blur_strength_layout = QHBoxLayout()
        blur_strength_layout.addWidget(QLabel("モザイク強度:"))
        self.blur_strength_slider = QSlider(Qt.Orientation.Horizontal)
        self.blur_strength_slider.setRange(1, 50)
        self.blur_strength_slider.setValue(DEFAULT_CONFIG["inference"]["blur_strength"])
        self.blur_strength_label = QLabel(str(DEFAULT_CONFIG["inference"]["blur_strength"]))
        self.blur_strength_slider.valueChanged.connect(
            lambda v: self.blur_strength_label.setText(str(v))
        )
        blur_strength_layout.addWidget(self.blur_strength_slider)
        blur_strength_layout.addWidget(self.blur_strength_label)
        layout.addLayout(blur_strength_layout)
        
        tile_size_layout = QHBoxLayout()
        tile_size_layout.addWidget(QLabel("タイルサイズ (ピクセル):"))
        self.tile_size_spin = QSpinBox()
        self.tile_size_spin.setRange(1, 100)
        self.tile_size_spin.setValue(10)
        tile_size_layout.addWidget(self.tile_size_spin)
        layout.addLayout(tile_size_layout)
        
        self.preserve_png_check = QCheckBox("PNGメタデータを保持")
        self.preserve_png_check.setChecked(True)
        layout.addWidget(self.preserve_png_check)
        
        return group
    
    def refresh_models(self):
        self.model_combo.clear()
        
        for model_file in MODELS_DIR.glob("**/*.pt"):
            if "best.pt" in model_file.name or "last.pt" in model_file.name:
                self.model_combo.addItem(
                    f"{model_file.parent.name}/{model_file.name}",
                    str(model_file)
                )
    
    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "画像を選択", "", "Image Files (*.jpg *.jpeg *.png)"
        )
        if file_path:
            self.current_image_path = Path(file_path)
            self.load_and_display_image(file_path)
    
    def load_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "フォルダを選択")
        if folder:
            self.file_list.clear()
            folder_path = Path(folder)
            
            for img_path in sorted(folder_path.glob("*.jpg")) + \
                           sorted(folder_path.glob("*.jpeg")) + \
                           sorted(folder_path.glob("*.png")):
                item = QListWidgetItem(img_path.name)
                item.setData(Qt.ItemDataRole.UserRole, str(img_path))
                self.file_list.addItem(item)
    
    def on_file_selected(self, item):
        file_path = item.data(Qt.ItemDataRole.UserRole)
        self.current_image_path = Path(file_path)
        self.load_and_display_image(file_path)
    
    def load_and_display_image(self, image_path):
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.current_image = image
        self.image_viewer.set_image(image)
        self.apply_mosaic_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
    
    def run_inference(self):
        if not self.current_image_path or self.model_combo.count() == 0:
            return
        
        model_path = self.model_combo.currentData()
        if not model_path:
            return
        
        self.inference_btn.setEnabled(False)
        
        self.inference_thread = InferenceThread(
            model_path,
            self.current_image_path,
            self.conf_spin.value(),
            self.iou_spin.value()
        )
        self.inference_thread.result_ready.connect(self.on_inference_complete)
        self.inference_thread.progress.connect(print)
        self.inference_thread.error.connect(print)
        self.inference_thread.start()
    
    def on_inference_complete(self, image, detections):
        self.current_image = image
        self.current_detections = detections
        self.inference_btn.setEnabled(True)
        self.apply_mosaic_btn.setEnabled(True)
        
        display_image = image.copy()
        for det in detections:
            mask = det['mask']
            overlay = display_image.copy()
            overlay[mask] = [255, 0, 0]
            display_image = cv2.addWeighted(display_image, 0.7, overlay, 0.3, 0)
        
        self.image_viewer.set_image(display_image)
    
    def apply_mosaic(self):
        if not self.current_detections:
            return
        
        from ..inference.mosaic import apply_mosaic_to_regions
        
        blur_type = self.blur_type_combo.currentText()
        strength = self.tile_size_spin.value() if blur_type == 'tile' else self.blur_strength_slider.value()
        
        self.processed_image = apply_mosaic_to_regions(
            self.current_image,
            self.current_detections,
            blur_type,
            strength
        )
        
        self.image_viewer.set_image(self.processed_image)
        self.save_btn.setEnabled(True)
    
    def save_image(self):
        if self.processed_image is None:
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "画像を保存", "", "Image Files (*.jpg *.png)"
        )
        if file_path:
            image_bgr = cv2.cvtColor(self.processed_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(file_path, image_bgr)
    
    def batch_process(self):
        if self.model_combo.count() == 0:
            return
        
        input_folder = QFileDialog.getExistingDirectory(self, "入力フォルダを選択")
        if not input_folder:
            return
            
        output_folder = QFileDialog.getExistingDirectory(self, "出力フォルダを選択")
        if not output_folder:
            return
        
        from pathlib import Path
        from ..inference.mosaic import batch_process_images
        
        input_path = Path(input_folder)
        image_paths = list(input_path.glob("*.png")) + list(input_path.glob("*.jpg")) + list(input_path.glob("*.jpeg"))
        
        if not image_paths:
            return
        
        model_path = self.model_combo.currentData()
        blur_type = self.blur_type_combo.currentText()
        strength = self.tile_size_spin.value() if blur_type == 'tile' else self.blur_strength_slider.value()
        
        from .batch_process_dialog import BatchProcessDialog
        
        config = {
            'confidence': self.conf_spin.value(),
            'iou': self.iou_spin.value(),
            'blur_type': blur_type,
            'strength': strength,
            'preserve_png_info': self.preserve_png_check.isChecked()
        }
        
        dialog = BatchProcessDialog(
            image_paths,
            model_path,
            output_folder,
            config,
            self
        )
        dialog.exec()