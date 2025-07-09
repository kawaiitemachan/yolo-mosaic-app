from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                               QLabel, QSlider, QComboBox, QListWidget,
                               QSplitter, QFileDialog, QListWidgetItem,
                               QGroupBox, QDoubleSpinBox, QCheckBox, QSpinBox,
                               QLineEdit, QMessageBox, QProgressBar, QScrollArea)
from PySide6.QtCore import Qt, QThread, Signal, QSettings
from PySide6.QtGui import QImage, QPixmap, QPainter
import cv2
import numpy as np
from pathlib import Path

from ..config import MODELS_DIR, DEFAULT_CONFIG
from ..utils.device_utils import get_device

def expand_bbox(x1, y1, x2, y2, expansion_percent, image_width, image_height):
    """
    バウンディングボックスを指定された割合で拡張する
    
    Args:
        x1, y1, x2, y2: バウンディングボックスの座標
        expansion_percent: 拡張率（%）
        image_width: 画像の幅
        image_height: 画像の高さ
    
    Returns:
        拡張されたバウンディングボックスの座標 (x1, y1, x2, y2)
    """
    if expansion_percent <= 0:
        return x1, y1, x2, y2
    
    # ボックスの中心と現在のサイズを計算
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    width = x2 - x1
    height = y2 - y1
    
    # 拡張率を適用
    expansion_factor = 1 + (expansion_percent / 100)
    new_width = width * expansion_factor
    new_height = height * expansion_factor
    
    # 新しい座標を計算
    new_x1 = int(center_x - new_width / 2)
    new_y1 = int(center_y - new_height / 2)
    new_x2 = int(center_x + new_width / 2)
    new_y2 = int(center_y + new_height / 2)
    
    # 画像の境界内に制限
    new_x1 = max(0, new_x1)
    new_y1 = max(0, new_y1)
    new_x2 = min(image_width, new_x2)
    new_y2 = min(image_height, new_y2)
    
    return new_x1, new_y1, new_x2, new_y2

class InferenceThread(QThread):
    result_ready = Signal(np.ndarray, list, str)  # 画像パスを追加
    progress = Signal(str)
    progress_update = Signal(int, int, str)  # 現在の番号, 総数, ファイル名
    error = Signal(str)
    
    def __init__(self, model_path, image_paths, confidence, blur_type, strength, output_dir=None, mask_expansion=2, selected_classes=None, use_bbox=False):
        super().__init__()
        self.model_path = model_path
        self.image_paths = image_paths if isinstance(image_paths, list) else [image_paths]
        self.confidence = confidence
        self.iou = 0.9  # 固定値：検出の重複を最小限に抑える
        self.blur_type = blur_type
        self.strength = strength
        self.output_dir = output_dir
        self.mask_expansion = mask_expansion
        self.selected_classes = selected_classes if selected_classes is not None else set()
        self.use_bbox = use_bbox
        
    def run(self):
        try:
            from ultralytics import YOLO
            from ..inference.mosaic import apply_mosaic_to_regions
            
            self.progress.emit("モデルを読み込み中...")
            model = YOLO(self.model_path)
            
            device, _ = get_device()
            
            total_images = len(self.image_paths)
            for idx, image_path in enumerate(self.image_paths):
                file_name = Path(image_path).name
                self.progress.emit(f"処理中 ({idx + 1}/{total_images}): {file_name}")
                self.progress_update.emit(idx + 1, total_images, file_name)
                
                # 推論実行
                results = model(
                    str(image_path),
                    conf=self.confidence,
                    iou=self.iou,
                    device=device
                )
                
                # 画像読み込み
                image = cv2.imread(str(image_path))
                if image is None:
                    self.error.emit(f"画像を読み込めません: {image_path}")
                    continue
                    
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # 検出結果を処理
                detections = []
                for r in results:
                    if self.use_bbox and r.boxes is not None:
                        # バウンディングボックスモード
                        for i, box in enumerate(r.boxes.xyxy):
                            cls = int(r.boxes.cls[i])
                            
                            # 選択されたクラスのみを処理
                            if self.selected_classes and cls not in self.selected_classes:
                                continue
                            
                            # バウンディングボックスを拡張
                            x1, y1, x2, y2 = map(int, box)
                            x1, y1, x2, y2 = expand_bbox(
                                x1, y1, x2, y2, 
                                self.mask_expansion,
                                image.shape[1], image.shape[0]
                            )
                            
                            # バウンディングボックスからマスクを作成
                            mask = np.zeros((image.shape[0], image.shape[1]), dtype=bool)
                            mask[y1:y2, x1:x2] = True
                            
                            conf = float(r.boxes.conf[i])
                            label = model.names[cls]
                            
                            detections.append({
                                'mask': mask,
                                'label': label,
                                'confidence': conf
                            })
                    elif not self.use_bbox and r.masks is not None:
                        # セグメンテーションモード（従来の処理）
                        for i, mask in enumerate(r.masks.data):
                            cls = int(r.boxes.cls[i])
                            
                            # 選択されたクラスのみを処理（選択されていない場合は全クラスを処理）
                            if self.selected_classes and cls not in self.selected_classes:
                                continue
                            
                            mask_np = mask.cpu().numpy()
                            mask_resized = cv2.resize(mask_np, (image.shape[1], image.shape[0]))
                            
                            conf = float(r.boxes.conf[i])
                            label = model.names[cls]
                            
                            detections.append({
                                'mask': mask_resized > 0.5,
                                'label': label,
                                'confidence': conf
                            })
                
                # 出力ディレクトリが指定されている場合
                if self.output_dir:
                    # サブフォルダを作成
                    detected_dir = Path(self.output_dir) / "検出あり"
                    undetected_dir = Path(self.output_dir) / "未検出"
                    detected_dir.mkdir(parents=True, exist_ok=True)
                    undetected_dir.mkdir(parents=True, exist_ok=True)
                    
                    if detections:
                        # 検出がある場合: モザイク適用して保存
                        processed_image = apply_mosaic_to_regions(
                            image,
                            detections,
                            self.blur_type,
                            self.strength,
                            self.mask_expansion
                        )
                        
                        output_path = detected_dir / Path(image_path).name
                        image_bgr = cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(str(output_path), image_bgr)
                        self.progress.emit(f"保存[検出あり]: {output_path.name}")
                    else:
                        # 検出がない場合: 元画像をコピー
                        output_path = undetected_dir / Path(image_path).name
                        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(str(output_path), image_bgr)
                        self.progress.emit(f"保存[未検出]: {output_path.name}")
                
                # 結果を送信
                self.result_ready.emit(image, detections, str(image_path))
            
            self.progress.emit("全ての処理が完了しました")
            
            # 処理結果のサマリーを出力
            if self.output_dir:
                detected_dir = Path(self.output_dir) / "検出あり"
                undetected_dir = Path(self.output_dir) / "未検出"
                detected_count = len(list(detected_dir.glob("*.*"))) if detected_dir.exists() else 0
                undetected_count = len(list(undetected_dir.glob("*.*"))) if undetected_dir.exists() else 0
                self.progress.emit(f"処理完了: 検出あり {detected_count}枚, 未検出 {undetected_count}枚")
            
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
        self.settings = QSettings("YoloMosaicApp", "Inference")
        self.current_image = None
        self.current_image_path = None
        self.current_detections = []
        self.folder_mode = False
        self.folder_path = None
        self.image_files = []
        self.processed_image = None
        self.class_checkboxes = {}  # クラス名とチェックボックスの対応
        self.model_classes = {}  # モデルのクラス情報
        self.selected_classes = set()  # 選択されたクラスID
        self.no_model_label = None  # 初期化
        self.init_ui()
        self.load_settings()
        
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
        
        # クラス選択グループを追加
        self.class_group = self.create_class_selection_group()
        layout.addWidget(self.class_group)
        
        mosaic_group = self.create_mosaic_group()
        layout.addWidget(mosaic_group)
        
        self.load_image_btn = QPushButton("画像を選択")
        self.load_image_btn.clicked.connect(self.load_image)
        layout.addWidget(self.load_image_btn)
        
        self.load_folder_btn = QPushButton("フォルダを選択")
        self.load_folder_btn.clicked.connect(self.load_folder)
        layout.addWidget(self.load_folder_btn)
        
        self.file_list = QListWidget()
        self.file_list.itemClicked.connect(self.on_file_selected)
        layout.addWidget(self.file_list)
        
        # 出力先フォルダ設定
        output_group = QGroupBox("出力先設定")
        output_layout = QVBoxLayout(output_group)
        
        path_layout = QHBoxLayout()
        self.output_path_edit = QLineEdit()
        self.output_path_edit.setPlaceholderText("出力先フォルダを選択...")
        self.output_path_edit.setReadOnly(True)
        path_layout.addWidget(self.output_path_edit)
        
        self.browse_output_btn = QPushButton("参照...")
        self.browse_output_btn.clicked.connect(self.browse_output_folder)
        path_layout.addWidget(self.browse_output_btn)
        
        output_layout.addLayout(path_layout)
        layout.addWidget(output_group)
        
        # 進捗表示
        progress_group = QGroupBox("処理状況")
        progress_layout = QVBoxLayout(progress_group)
        
        self.progress_label = QLabel("待機中...")
        self.progress_label.setWordWrap(True)
        progress_layout.addWidget(self.progress_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("QLabel { color: #666; font-size: 12px; }")
        progress_layout.addWidget(self.status_label)
        
        layout.addWidget(progress_group)
        
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
        
        # 並列処理数の設定
        parallel_layout = QHBoxLayout()
        parallel_layout.addWidget(QLabel("並列処理数:"))
        self.parallel_spin = QSpinBox()
        self.parallel_spin.setRange(1, 8)
        self.parallel_spin.setValue(1)
        self.parallel_spin.setToolTip("同時に処理する画像数（1=順次処理）")
        self.parallel_spin.valueChanged.connect(self.save_settings)
        parallel_layout.addWidget(self.parallel_spin)
        layout.addLayout(parallel_layout)
        
        layout.addStretch()
        
        # UI作成後にモデルリストを更新し、イベントを接続
        self.model_combo.currentIndexChanged.connect(self.on_model_changed)
        self.refresh_models()
        
        # 最初のモデルのクラスリストを更新（UIが全て作成された後）
        if self.model_combo.count() > 0:
            self.on_model_changed(0)
            
        return widget
    
    def create_model_group(self):
        group = QGroupBox("モデル設定")
        layout = QVBoxLayout(group)
        
        self.model_combo = QComboBox()
        # 初期化中は接続しない
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
        self.conf_spin.valueChanged.connect(self.save_settings)
        conf_layout.addWidget(self.conf_spin)
        layout.addLayout(conf_layout)
        
        return group
    
    def create_mosaic_group(self):
        group = QGroupBox("モザイク設定")
        layout = QVBoxLayout(group)
        
        blur_type_layout = QHBoxLayout()
        blur_type_layout.addWidget(QLabel("モザイクタイプ:"))
        self.blur_type_combo = QComboBox()
        self.blur_type_combo.addItems(["gaussian", "pixelate", "blur", "black", "white", "tile"])
        self.blur_type_combo.currentTextChanged.connect(self.save_settings)
        blur_type_layout.addWidget(self.blur_type_combo)
        layout.addLayout(blur_type_layout)
        
        blur_strength_layout = QHBoxLayout()
        blur_strength_layout.addWidget(QLabel("モザイク強度:"))
        self.blur_strength_slider = QSlider(Qt.Orientation.Horizontal)
        self.blur_strength_slider.setRange(1, 50)
        self.blur_strength_slider.setValue(DEFAULT_CONFIG["inference"]["blur_strength"])
        self.blur_strength_label = QLabel(str(DEFAULT_CONFIG["inference"]["blur_strength"]))
        self.blur_strength_slider.valueChanged.connect(self.on_blur_strength_changed)
        blur_strength_layout.addWidget(self.blur_strength_slider)
        blur_strength_layout.addWidget(self.blur_strength_label)
        layout.addLayout(blur_strength_layout)
        
        tile_size_layout = QHBoxLayout()
        tile_size_layout.addWidget(QLabel("タイルサイズ (ピクセル):"))
        self.tile_size_spin = QSpinBox()
        self.tile_size_spin.setRange(1, 100)
        self.tile_size_spin.setValue(10)
        self.tile_size_spin.valueChanged.connect(self.save_settings)
        tile_size_layout.addWidget(self.tile_size_spin)
        layout.addLayout(tile_size_layout)
        
        self.preserve_png_check = QCheckBox("PNGメタデータを保持")
        self.preserve_png_check.setChecked(True)
        self.preserve_png_check.stateChanged.connect(self.save_settings)
        layout.addWidget(self.preserve_png_check)
        
        # バウンディングボックスモード
        self.use_bbox_check = QCheckBox("バウンディングボックスで処理")
        self.use_bbox_check.setChecked(False)
        self.use_bbox_check.setToolTip("セグメンテーションマスクの代わりにバウンディングボックスを使用")
        self.use_bbox_check.stateChanged.connect(self.on_bbox_mode_changed)
        layout.addWidget(self.use_bbox_check)
        
        # マスク拡張率設定
        mask_expand_layout = QHBoxLayout()
        mask_expand_layout.addWidget(QLabel("マスク拡張率 (%):"))
        self.mask_expand_spin = QSpinBox()
        self.mask_expand_spin.setRange(0, 10)
        self.mask_expand_spin.setValue(2)  # デフォルト2%
        self.mask_expand_spin.setSuffix("%")
        self.mask_expand_spin.setToolTip("検出されたマスクを拡張する割合")
        self.mask_expand_spin.valueChanged.connect(self.save_settings)
        mask_expand_layout.addWidget(self.mask_expand_spin)
        layout.addLayout(mask_expand_layout)
        
        return group
    
    def create_class_selection_group(self):
        """クラス選択用のグループボックスを作成"""
        group = QGroupBox("処理対象クラス")
        layout = QVBoxLayout(group)
        
        # スクロールエリアを作成（クラスが多い場合のため）
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMaximumHeight(200)
        
        # スクロールエリア内のウィジェット
        self.class_widget = QWidget()
        self.class_layout = QVBoxLayout(self.class_widget)
        
        # 全選択/全解除ボタン
        button_layout = QHBoxLayout()
        self.select_all_btn = QPushButton("全選択")
        self.select_all_btn.clicked.connect(self.select_all_classes)
        self.deselect_all_btn = QPushButton("全解除")
        self.deselect_all_btn.clicked.connect(self.deselect_all_classes)
        button_layout.addWidget(self.select_all_btn)
        button_layout.addWidget(self.deselect_all_btn)
        layout.addLayout(button_layout)
        
        # 初期状態では「モデルを選択してください」と表示
        self.no_model_label = QLabel("モデルを選択してください")
        self.class_layout.addWidget(self.no_model_label)
        
        scroll.setWidget(self.class_widget)
        layout.addWidget(scroll)
        
        return group
    
    def on_model_changed(self, index):
        """モデルが変更されたときの処理"""
        if index < 0:
            return
            
        model_path = self.model_combo.currentData()
        if not model_path:
            return
            
        # モデルからクラス情報を取得
        self.load_model_classes(model_path)
        
        # クラス選択UIを更新
        self.update_class_selection_ui()
        
        # 設定を保存
        self.save_settings()
    
    def load_model_classes(self, model_path):
        """モデルからクラス情報を取得"""
        try:
            from ultralytics import YOLO
            model = YOLO(model_path)
            self.model_classes = model.names  # {0: 'person', 1: 'bicycle', ...}
        except Exception as e:
            print(f"モデルクラスの取得エラー: {e}")
            self.model_classes = {}
    
    def update_class_selection_ui(self):
        """クラス選択UIを更新"""
        # class_layoutが初期化されているか確認
        if not hasattr(self, 'class_layout'):
            return
            
        # 既存のチェックボックスをクリア
        for checkbox in self.class_checkboxes.values():
            checkbox.deleteLater()
        self.class_checkboxes.clear()
        
        if self.no_model_label:
            self.no_model_label.deleteLater()
            self.no_model_label = None
        
        # 新しいチェックボックスを作成
        for class_id, class_name in sorted(self.model_classes.items()):
            checkbox = QCheckBox(f"{class_name} (ID: {class_id})")
            checkbox.setChecked(True)  # デフォルトは全選択
            checkbox.stateChanged.connect(lambda state, cid=class_id: self.on_class_selection_changed(cid, state))
            self.class_layout.addWidget(checkbox)
            self.class_checkboxes[class_id] = checkbox
        
        # 設定から選択状態を復元
        self.restore_class_selection()
    
    def on_class_selection_changed(self, class_id, state):
        """クラスの選択状態が変更されたときの処理"""
        if state == Qt.CheckState.Checked.value:
            self.selected_classes.add(class_id)
        else:
            self.selected_classes.discard(class_id)
        self.save_settings()
    
    def select_all_classes(self):
        """全クラスを選択"""
        for class_id, checkbox in self.class_checkboxes.items():
            checkbox.setChecked(True)
            self.selected_classes.add(class_id)
        self.save_settings()
    
    def deselect_all_classes(self):
        """全クラスを解除"""
        for class_id, checkbox in self.class_checkboxes.items():
            checkbox.setChecked(False)
            self.selected_classes.discard(class_id)
        self.save_settings()
    
    def refresh_models(self):
        self.model_combo.clear()
        print(f"Searching for models in: {MODELS_DIR}")
        
        model_count = 0
        # モデルフォルダ（データセット）ごとに集約
        model_dirs = {}
        
        for model_file in MODELS_DIR.glob("**/*.pt"):
            if "best.pt" in model_file.name or "last.pt" in model_file.name:
                # weightsフォルダの親ディレクトリ（データセット名）を取得
                if model_file.parent.name == "weights":
                    dataset_name = model_file.parent.parent.name
                else:
                    dataset_name = model_file.parent.name
                
                if dataset_name not in model_dirs:
                    model_dirs[dataset_name] = {"best": None, "last": None}
                
                if "best.pt" in model_file.name:
                    model_dirs[dataset_name]["best"] = str(model_file)
                elif "last.pt" in model_file.name:
                    model_dirs[dataset_name]["last"] = str(model_file)
        
        # データセット名でコンボボックスに追加
        for dataset_name, paths in sorted(model_dirs.items()):
            # best.ptを優先、なければlast.ptを使用
            model_path = paths["best"] if paths["best"] else paths["last"]
            if model_path:
                self.model_combo.addItem(dataset_name, model_path)
                model_count += 1
                print(f"Found model: {dataset_name} -> {model_path}")
        
        print(f"Total models found: {model_count}")
        
        # 最初のモデルが選択されたときにクラスリストを更新
        # ただし、class_layoutがまだ作成されていない場合はスキップ
        if self.model_combo.count() > 0 and hasattr(self, 'class_layout'):
            self.on_model_changed(0)
    
    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "画像を選択", "", "Image Files (*.jpg *.jpeg *.png)"
        )
        if file_path:
            self.folder_mode = False  # 単一画像モードに切り替え
            self.image_files = []
            self.load_and_display_image(file_path)
    
    def load_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "フォルダを選択")
        if folder:
            self.file_list.clear()
            self.folder_path = Path(folder)
            self.folder_mode = True
            
            # 画像ファイルを収集
            self.image_files = []
            for img_path in sorted(self.folder_path.glob("*.jpg")) + \
                           sorted(self.folder_path.glob("*.jpeg")) + \
                           sorted(self.folder_path.glob("*.png")):
                item = QListWidgetItem(img_path.name)
                item.setData(Qt.ItemDataRole.UserRole, str(img_path))
                self.file_list.addItem(item)
                self.image_files.append(str(img_path))
            
            # フォルダ情報を表示
            if self.image_files:
                from PySide6.QtWidgets import QMessageBox
                QMessageBox.information(
                    self, 
                    "フォルダ選択", 
                    f"{len(self.image_files)} 枚の画像が見つかりました\n"
                    f"推論実行ボタンで全ての画像を処理します"
                )
    
    def on_file_selected(self, item):
        file_path = item.data(Qt.ItemDataRole.UserRole)
        self.load_and_display_image(file_path)
    
    def browse_output_folder(self):
        """出力先フォルダを選択"""
        folder = QFileDialog.getExistingDirectory(self, "出力先フォルダを選択")
        if folder:
            self.output_path_edit.setText(folder)
            self.save_settings()
    
    def load_and_display_image(self, image_path):
        # パスを保存（重要！）
        self.current_image_path = Path(image_path)
        print(f"Loading image: {self.current_image_path}")
        
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Failed to load image: {image_path}")
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "エラー", f"画像を読み込めませんでした: {image_path}")
            return
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.current_image = image
        self.image_viewer.set_image(image)
        self.apply_mosaic_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        print(f"Image loaded successfully: {image.shape}")
    
    def run_inference(self):
        # モデルチェック
        if self.model_combo.count() == 0:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "警告", "モデルが見つかりません。学習済みモデルを作成してください")
            return
        
        model_path = self.model_combo.currentData()
        if not model_path:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "警告", "有効なモデルを選択してください")
            return
        
        # フォルダモードの場合
        if self.folder_mode and self.image_files:
            print(f"フォルダモード: {len(self.image_files)} 枚の画像を処理")
            
            # 出力フォルダの確認
            output_folder = self.output_path_edit.text()
            if not output_folder:
                from PySide6.QtWidgets import QMessageBox
                QMessageBox.warning(self, "警告", "出力先フォルダを指定してください")
                return
            
            # モザイク設定を取得
            blur_type = self.blur_type_combo.currentText()
            strength = self.tile_size_spin.value() if blur_type == 'tile' else self.blur_strength_slider.value()
            
            self.inference_btn.setEnabled(False)
            self.inference_thread = InferenceThread(
                model_path,
                self.image_files,  # リストを渡す
                self.conf_spin.value(),
                blur_type,
                strength,
                output_folder,
                self.mask_expand_spin.value(),
                self.selected_classes,
                self.use_bbox_check.isChecked()
            )
        
        # 単一画像モード
        elif self.current_image_path:
            print(f"単一画像モード: {self.current_image_path}")
            
            # モザイク設定を取得（単一画像でも必要）
            blur_type = self.blur_type_combo.currentText()
            strength = self.tile_size_spin.value() if blur_type == 'tile' else self.blur_strength_slider.value()
            
            # 出力フォルダを取得（単一画像でも保存可能）
            output_folder = self.output_path_edit.text() if self.output_path_edit.text() else None
            
            self.inference_btn.setEnabled(False)
            self.inference_thread = InferenceThread(
                model_path,
                str(self.current_image_path),
                self.conf_spin.value(),
                blur_type,
                strength,
                output_folder,  # 単一画像でも出力フォルダを使用
                self.mask_expand_spin.value(),
                self.selected_classes,
                self.use_bbox_check.isChecked()
            )
        
        else:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "警告", "画像またはフォルダを選択してください")
            return
        
        # 共通のシグナル接続
        self.inference_thread.result_ready.connect(self.on_inference_complete)
        self.inference_thread.progress.connect(self.on_inference_progress)
        self.inference_thread.progress_update.connect(self.on_progress_update)
        self.inference_thread.error.connect(self.on_inference_error)
        self.inference_thread.finished.connect(self.on_inference_finished)
        self.inference_thread.start()
        
        # UI制御
        self.set_ui_enabled(False)
        self.progress_bar.setValue(0)
        self.progress_label.setText("処理を開始しています...")
        print("Inference thread started")
    
    def on_inference_complete(self, image, detections, image_path):
        # フォルダモードの場合は画面更新をスキップ（最後の画像のみ表示）
        if not self.folder_mode or (self.image_files and image_path == self.image_files[-1]):
            self.current_image = image
            self.current_detections = detections
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
            strength,
            self.mask_expand_spin.value()
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
    
    def on_inference_finished(self):
        """推論スレッドが終了したときの処理"""
        print("推論スレッド終了")
        self.set_ui_enabled(True)
        self.progress_label.setText("処理完了")
        self.progress_bar.setValue(100)
        
        if self.folder_mode:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.information(self, "完了", "全ての画像の処理が完了しました")
    
    
    def on_inference_progress(self, message):
        """推論の進捗メッセージを処理"""
        print(f"推論進捗: {message}")
        self.status_label.setText(message)
    
    def on_progress_update(self, current, total, filename):
        """進捗バーの更新"""
        progress = int((current / total) * 100) if total > 0 else 0
        self.progress_bar.setValue(progress)
        self.progress_label.setText(f"処理中: {filename} ({current}/{total})")
    
    def on_inference_error(self, error_message):
        """推論エラーを処理"""
        print(f"推論エラー: {error_message}")
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.critical(self, "推論エラー", error_message)
        self.set_ui_enabled(True)
        self.progress_label.setText("エラーが発生しました")
        self.status_label.setText(error_message)
    
    def set_ui_enabled(self, enabled):
        """処理中のUI制御"""
        self.inference_btn.setEnabled(enabled)
        self.model_combo.setEnabled(enabled)
        self.load_image_btn.setEnabled(enabled)
        self.load_folder_btn.setEnabled(enabled)
        self.browse_output_btn.setEnabled(enabled)
        self.file_list.setEnabled(enabled)
        self.apply_mosaic_btn.setEnabled(enabled and bool(self.current_detections))
        self.save_btn.setEnabled(enabled and hasattr(self, 'processed_image') and self.processed_image is not None)
    
    def on_blur_strength_changed(self, value):
        """ブラー強度スライダーの値が変更されたとき"""
        self.blur_strength_label.setText(str(value))
        self.save_settings()
    
    def on_bbox_mode_changed(self, state):
        """バウンディングボックスモードの切り替え"""
        # バウンディングボックスモードでも拡張率は有効
        if self.use_bbox_check.isChecked():
            self.mask_expand_spin.setToolTip("バウンディングボックスを拡張する割合")
        else:
            self.mask_expand_spin.setToolTip("検出されたマスクを拡張する割合")
        self.save_settings()
    
    def save_settings(self):
        """設定を保存"""
        self.settings.setValue("confidence", self.conf_spin.value())
        self.settings.setValue("blur_type", self.blur_type_combo.currentText())
        self.settings.setValue("blur_strength", self.blur_strength_slider.value())
        self.settings.setValue("tile_size", self.tile_size_spin.value())
        self.settings.setValue("preserve_png", self.preserve_png_check.isChecked())
        self.settings.setValue("parallel_count", self.parallel_spin.value())
        self.settings.setValue("output_folder", self.output_path_edit.text())
        self.settings.setValue("mask_expansion", self.mask_expand_spin.value())
        self.settings.setValue("use_bbox", self.use_bbox_check.isChecked())
        
        # 選択されたクラスを保存（モデルごとに保存）
        if self.model_combo.currentData():
            model_name = self.model_combo.currentText()
            selected_list = list(self.selected_classes)
            self.settings.setValue(f"selected_classes_{model_name}", selected_list)
    
    def load_settings(self):
        """設定を読み込み"""
        # 信頼度閾値
        confidence = self.settings.value("confidence", DEFAULT_CONFIG["inference"]["confidence"], type=float)
        self.conf_spin.setValue(confidence)
        
        # モザイクタイプ
        blur_type = self.settings.value("blur_type", DEFAULT_CONFIG["inference"]["blur_type"])
        index = self.blur_type_combo.findText(blur_type)
        if index >= 0:
            self.blur_type_combo.setCurrentIndex(index)
        
        # モザイク強度
        blur_strength = self.settings.value("blur_strength", DEFAULT_CONFIG["inference"]["blur_strength"], type=int)
        self.blur_strength_slider.setValue(blur_strength)
        self.blur_strength_label.setText(str(blur_strength))
        
        # タイルサイズ
        tile_size = self.settings.value("tile_size", 10, type=int)
        self.tile_size_spin.setValue(tile_size)
        
        # PNGメタデータ保持
        preserve_png = self.settings.value("preserve_png", True, type=bool)
        self.preserve_png_check.setChecked(preserve_png)
        
        # 並列処理数
        parallel_count = self.settings.value("parallel_count", 1, type=int)
        self.parallel_spin.setValue(parallel_count)
        
        # 出力フォルダ
        output_folder = self.settings.value("output_folder", "")
        self.output_path_edit.setText(output_folder)
        
        # マスク拡張率
        mask_expansion = self.settings.value("mask_expansion", 2, type=int)
        self.mask_expand_spin.setValue(mask_expansion)
        
        # バウンディングボックスモード
        use_bbox = self.settings.value("use_bbox", False, type=bool)
        self.use_bbox_check.setChecked(use_bbox)
        # ツールチップを更新
        if use_bbox:
            self.mask_expand_spin.setToolTip("バウンディングボックスを拡張する割合")
        else:
            self.mask_expand_spin.setToolTip("検出されたマスクを拡張する割合")
    
    def restore_class_selection(self):
        """保存されたクラス選択状態を復元"""
        if not self.model_combo.currentData():
            return
            
        model_name = self.model_combo.currentText()
        saved_classes = self.settings.value(f"selected_classes_{model_name}", None)
        
        if saved_classes is not None:
            # 保存された選択状態を復元
            self.selected_classes = set(saved_classes)
            for class_id, checkbox in self.class_checkboxes.items():
                checkbox.setChecked(class_id in self.selected_classes)
        else:
            # デフォルトは全選択
            self.selected_classes = set(self.model_classes.keys())
            for checkbox in self.class_checkboxes.values():
                checkbox.setChecked(True)