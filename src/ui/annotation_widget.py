from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                               QListWidget, QComboBox, QLabel, QSlider,
                               QSplitter, QFileDialog, QListWidgetItem,
                               QMessageBox, QDialog, QInputDialog)
from PySide6.QtCore import Qt, Signal, QPointF
from PySide6.QtGui import QPainter, QPen, QColor, QPolygonF, QImage, QPixmap

import cv2
import numpy as np
from pathlib import Path
import yaml

from ..config import IMAGES_DIR, DEFAULT_CONFIG

class ImageCanvas(QWidget):
    polygon_completed = Signal(list)
    
    def __init__(self):
        super().__init__()
        self.image = None
        self.pixmap = None
        self.current_polygon = []
        self.polygons = []
        self.current_label = "object"
        self.drawing = False
        self.scale = 1.0
        self.offset = QPointF(0, 0)
        self.class_colors = {}  # クラスごとの色を動的に管理
        
    def load_image(self, image_path):
        self.image = QImage(str(image_path))
        self.pixmap = QPixmap.fromImage(self.image)
        self.fit_image_to_widget()
        self.current_polygon = []
        self.polygons = []
        self.update()
        
    def fit_image_to_widget(self):
        if self.pixmap:
            widget_size = self.size()
            image_size = self.pixmap.size()
            
            scale_x = widget_size.width() / image_size.width()
            scale_y = widget_size.height() / image_size.height()
            self.scale = min(scale_x, scale_y, 1.0)
            
            scaled_width = image_size.width() * self.scale
            scaled_height = image_size.height() * self.scale
            
            self.offset = QPointF(
                (widget_size.width() - scaled_width) / 2,
                (widget_size.height() - scaled_height) / 2
            )
    
    def paintEvent(self, event):
        if not self.pixmap:
            return
            
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        painter.translate(self.offset)
        painter.scale(self.scale, self.scale)
        painter.drawPixmap(0, 0, self.pixmap)
        
        # クラス色が設定されていない場合はデフォルトを使用
        if not self.class_colors:
            colors = DEFAULT_CONFIG["annotation"]["colors"]
        else:
            colors = self.class_colors
        
        for polygon_data in self.polygons:
            polygon = polygon_data["points"]
            label = polygon_data["label"]
            color = QColor(colors.get(label, "#FF0000"))
            
            pen = QPen(color, 2)
            painter.setPen(pen)
            
            if len(polygon) > 1:
                for i in range(len(polygon)):
                    p1 = polygon[i]
                    p2 = polygon[(i + 1) % len(polygon)]
                    painter.drawLine(p1, p2)
                    
            for point in polygon:
                painter.setBrush(color)
                painter.drawEllipse(point, 3, 3)
        
        if self.current_polygon:
            color = QColor(colors.get(self.current_label, "#FF0000"))
            pen = QPen(color, 2)
            pen.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(pen)
            
            if len(self.current_polygon) > 1:
                for i in range(len(self.current_polygon) - 1):
                    painter.drawLine(self.current_polygon[i], self.current_polygon[i + 1])
            
            for point in self.current_polygon:
                painter.setBrush(color)
                painter.drawEllipse(point, 3, 3)
    
    def mousePressEvent(self, event):
        if not self.pixmap or event.button() != Qt.MouseButton.LeftButton:
            return
            
        pos = (event.position() - self.offset) / self.scale
        
        if 0 <= pos.x() <= self.pixmap.width() and 0 <= pos.y() <= self.pixmap.height():
            self.current_polygon.append(pos)
            self.update()
    
    def mouseDoubleClickEvent(self, event):
        if len(self.current_polygon) >= 3:
            self.complete_polygon()
    
    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Return and len(self.current_polygon) >= 3:
            self.complete_polygon()
        elif event.key() == Qt.Key.Key_Escape:
            self.current_polygon = []
            self.update()
    
    def complete_polygon(self):
        if len(self.current_polygon) >= 3:
            polygon_data = {
                "points": self.current_polygon.copy(),
                "label": self.current_label
            }
            self.polygons.append(polygon_data)
            self.polygon_completed.emit(self.current_polygon)
            self.current_polygon = []
            self.update()
    
    def set_label(self, label):
        self.current_label = label
    
    def set_class_colors(self, classes):
        """クラスごとの色を設定"""
        # 既存のデフォルト色
        default_colors = DEFAULT_CONFIG["annotation"]["colors"]
        
        # 視覚的に判別しやすい色のリスト（色相環上で離れた色を選択）
        distinctive_colors = [
            # 基本色
            "#FF0000",  # 赤
            "#0000FF",  # 青
            "#00FF00",  # 緑
            "#FFFF00",  # 黄色
            "#FF00FF",  # マゼンタ
            "#00FFFF",  # シアン
            "#FF8800",  # オレンジ
            "#8800FF",  # 紫
            
            # 明度を変えた基本色
            "#CC0000",  # 暗い赤
            "#0000CC",  # 暗い青
            "#00CC00",  # 暗い緑
            "#CCCC00",  # 暗い黄色
            "#CC00CC",  # 暗いマゼンタ
            "#00CCCC",  # 暗いシアン
            "#CC6600",  # 暗いオレンジ
            "#6600CC",  # 暗い紫
            
            # パステルカラー（背景から目立つように調整）
            "#FF6666",  # ライトレッド
            "#6666FF",  # ライトブルー
            "#66FF66",  # ライトグリーン
            "#FFFF66",  # ライトイエロー
            "#FF66FF",  # ライトマゼンタ
            "#66FFFF",  # ライトシアン
            "#FFB366",  # ライトオレンジ
            "#B366FF",  # ライトパープル
        ]
        
        self.class_colors = {}
        used_colors = set()
        
        # まず、デフォルトの色を使用
        for class_name in classes:
            if class_name in default_colors:
                self.class_colors[class_name] = default_colors[class_name]
                used_colors.add(default_colors[class_name])
        
        # 残りのクラスに対して、使用されていない色から割り当て
        available_colors = [c for c in distinctive_colors if c not in used_colors]
        color_index = 0
        
        for class_name in classes:
            if class_name not in self.class_colors:
                if color_index < len(available_colors):
                    self.class_colors[class_name] = available_colors[color_index]
                else:
                    # 全ての色を使い切った場合は、最初から再利用
                    self.class_colors[class_name] = available_colors[color_index % len(available_colors)]
                color_index += 1
    
    def clear_polygons(self):
        self.polygons = []
        self.current_polygon = []
        self.update()
    
    def undo_last_polygon(self):
        if self.polygons:
            self.polygons.pop()
            self.update()

class AnnotationWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.dataset_path = None
        self.images_folder = None
        self.current_split = "train"  # train or valid
        self.init_ui()
        self.current_image_path = None
    
    def closeEvent(self, event):
        """ウィジェットが閉じられる時に自動保存"""
        if self.current_image_path and self.canvas.polygons:
            self.save_annotations(silent=True)
        event.accept()
        
    def init_ui(self):
        layout = QHBoxLayout(self)
        
        self.canvas = ImageCanvas()
        self.canvas.polygon_completed.connect(self.on_polygon_completed)
        left_panel = self.create_left_panel()
        
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(self.canvas)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        
        layout.addWidget(splitter)
    
    def create_left_panel(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # データセット選択ボタン
        select_dataset_btn = QPushButton("データセット選択")
        select_dataset_btn.clicked.connect(self.select_dataset)
        layout.addWidget(select_dataset_btn)
        
        # データセット情報表示
        self.dataset_info_label = QLabel("データセット未選択")
        self.dataset_info_label.setWordWrap(True)
        layout.addWidget(self.dataset_info_label)
        
        # train/valid切り替え
        layout.addWidget(QLabel("データ分割:"))
        self.split_combo = QComboBox()
        self.split_combo.addItems(["train (訓練用)", "valid (検証用)"])
        self.split_combo.currentIndexChanged.connect(self.on_split_changed)
        layout.addWidget(self.split_combo)
        
        layout.addWidget(QLabel("画像リスト:"))
        self.image_list = QListWidget()
        self.image_list.itemClicked.connect(self.on_image_selected)
        layout.addWidget(self.image_list)
        
        layout.addWidget(QLabel("ラベル:"))
        self.label_combo = QComboBox()
        self.label_combo.currentTextChanged.connect(self.canvas.set_label)
        layout.addWidget(self.label_combo)
        
        # クラス管理ボタン
        class_management_layout = QHBoxLayout()
        add_class_btn = QPushButton("クラス追加")
        add_class_btn.clicked.connect(self.add_class)
        remove_class_btn = QPushButton("クラス削除")
        remove_class_btn.clicked.connect(self.remove_class)
        class_management_layout.addWidget(add_class_btn)
        class_management_layout.addWidget(remove_class_btn)
        layout.addLayout(class_management_layout)
        
        clear_btn = QPushButton("ポリゴンをクリア")
        clear_btn.clicked.connect(self.canvas.clear_polygons)
        layout.addWidget(clear_btn)
        
        undo_btn = QPushButton("最後のポリゴンを削除")
        undo_btn.clicked.connect(self.canvas.undo_last_polygon)
        layout.addWidget(undo_btn)
        
        save_btn = QPushButton("アノテーションを保存")
        save_btn.clicked.connect(self.save_annotations)
        layout.addWidget(save_btn)
        
        layout.addWidget(QLabel("操作方法:"))
        help_text = QLabel("左クリック: 点を追加\nダブルクリック/Enter: ポリゴン確定\nEsc: 現在のポリゴンをキャンセル")
        help_text.setWordWrap(True)
        layout.addWidget(help_text)
        
        layout.addStretch()
        return widget
    
    def select_dataset(self):
        """データセット選択ダイアログを表示"""
        from .dataset_dialog import DatasetSelectionDialog
        
        dialog = DatasetSelectionDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.dataset_path = dialog.dataset_path
            self.images_folder = dialog.images_path
            
            # データセット情報を更新
            self.update_dataset_info()
            
            # 画像リストを更新
            self.load_images_from_folder()
            
            # 自動保存通知
            self.notify_auto_save()
    
    def load_dataset(self, dataset_path):
        """指定されたデータセットを読み込む"""
        self.dataset_path = dataset_path
        
        # データセットから画像フォルダを推測
        # train/imagesがあればそれを使用
        train_images = dataset_path / "train" / "images"
        if train_images.exists():
            self.images_folder = train_images
        else:
            # データセット内の画像を探す
            for img_dir in dataset_path.rglob("images"):
                if img_dir.is_dir():
                    self.images_folder = img_dir
                    break
        
        # データセット情報を更新
        self.update_dataset_info()
        
        # 画像リストを更新
        if self.images_folder:
            self.load_images_from_folder()
        
        # 自動保存通知
        self.notify_auto_save()
    
    def notify_auto_save(self):
        """自動保存の通知を送る"""
        parent = self.parent()
        while parent and not hasattr(parent, 'show_save_notification'):
            parent = parent.parent()
        
        if parent and hasattr(parent, 'show_save_notification'):
            parent.show_save_notification("データセット設定を自動保存しました")
    
    def update_dataset_info(self):
        """データセット情報表示を更新"""
        if self.dataset_path:
            info_text = f"データセット: {self.dataset_path.name}\n"
            
            # data.yamlから情報を読み込む
            yaml_path = self.dataset_path / "data.yaml"
            if yaml_path.exists():
                try:
                    with open(yaml_path, 'r') as f:
                        data = yaml.safe_load(f)
                    classes = data.get('names', [])
                    info_text += f"クラス: {', '.join(classes)}"
                    
                    # ラベルコンボボックスを更新
                    self.update_label_combo(classes)
                except:
                    pass
            
            self.dataset_info_label.setText(info_text)
    
    def on_split_changed(self, index):
        """train/valid切り替え時の処理"""
        self.current_split = "train" if index == 0 else "valid"
        
        # 現在の画像のアノテーションを保存
        if self.current_image_path and self.canvas.polygons:
            self.save_annotations(silent=True)
    
    def load_images_from_folder(self):
        """指定されたフォルダから画像を読み込む"""
        if not self.images_folder:
            return
            
        self.image_list.clear()
        
        for img_path in sorted(self.images_folder.glob("*.jpg")) + sorted(self.images_folder.glob("*.png")):
            item = QListWidgetItem(img_path.name)
            item.setData(Qt.ItemDataRole.UserRole, str(img_path))
            self.image_list.addItem(item)
    
    def on_image_selected(self, item):
        # 現在の画像のアノテーションを自動保存
        if self.current_image_path and self.canvas.polygons:
            self.save_annotations(silent=True)
        
        image_path = item.data(Qt.ItemDataRole.UserRole)
        self.current_image_path = Path(image_path)
        self.canvas.load_image(image_path)
        
        # 既存のアノテーションを読み込む
        self.load_existing_annotations()
    
    def save_annotations(self, silent=False):
        if not self.current_image_path or not self.canvas.polygons:
            return
        
        if not self.dataset_path:
            if not silent:
                QMessageBox.warning(self, "警告", "データセットが選択されていません")
            return
            
        # データセット構造に従って保存
        from ..annotation.yolo_format import save_yolo_segmentation_to_dataset
        
        try:
            txt_path = save_yolo_segmentation_to_dataset(
                self.current_image_path,
                self.canvas.polygons,
                self.canvas.pixmap.width(),
                self.canvas.pixmap.height(),
                self.dataset_path,
                self.current_split
            )
            
            # data.yamlを更新
            self.update_data_yaml()
            
            if not silent:
                QMessageBox.information(self, "保存完了", f"アノテーションを保存しました:\n{txt_path}")
        except Exception as e:
            if not silent:
                QMessageBox.critical(self, "エラー", f"保存中にエラーが発生しました: {str(e)}")
    
    def load_existing_annotations(self):
        """既存のアノテーションファイルを読み込む"""
        if not self.current_image_path or not self.dataset_path:
            return
            
        from ..annotation.yolo_format import load_yolo_segmentation_from_dataset
        
        try:
            polygons = load_yolo_segmentation_from_dataset(
                self.current_image_path,
                self.dataset_path,
                self.current_split,
                self.canvas.pixmap.width(),
                self.canvas.pixmap.height()
            )
            self.canvas.polygons = polygons
            self.canvas.update()
        except:
            # アノテーションファイルが存在しない場合は無視
            pass
    
    def update_data_yaml(self):
        """data.yamlのクラス情報を更新"""
        if not self.dataset_path:
            return
            
        yaml_path = self.dataset_path / "data.yaml"
        
        try:
            # 既存のdata.yamlを読み込む
            with open(yaml_path, 'r') as f:
                data = yaml.safe_load(f)
            
            # 現在のクラス情報を取得（ここでは更新しない）
            # クラスの追加・削除はadd_class/remove_classメソッドで行う
            
        except Exception as e:
            print(f"data.yaml読み込みエラー: {e}")
    
    def on_polygon_completed(self, polygon):
        """ポリゴンが完成したときの処理"""
        # ポリゴン完成時に自動保存
        if hasattr(self, 'dataset_path') and self.dataset_path:
            self.save_annotations(silent=True)
    
    def update_label_combo(self, classes):
        """ラベルコンボボックスを更新"""
        current_label = self.label_combo.currentText()
        self.label_combo.clear()
        
        if classes:
            self.label_combo.addItems(classes)
            
            # Canvasの色設定を更新
            self.canvas.set_class_colors(classes)
            
            # 前の選択を復元
            if current_label in classes:
                self.label_combo.setCurrentText(current_label)
            elif classes:
                self.label_combo.setCurrentIndex(0)
                self.canvas.set_label(classes[0])
    
    def add_class(self):
        """クラスを追加"""
        if not self.dataset_path:
            QMessageBox.warning(self, "警告", "先にデータセットを選択してください")
            return
        
        # クラス名入力ダイアログ
        class_name, ok = QInputDialog.getText(
            self, 
            "クラス追加", 
            "新しいクラス名を入力してください:",
            text=""
        )
        
        if ok and class_name:
            # 既存のクラスを取得
            yaml_path = self.dataset_path / "data.yaml"
            if yaml_path.exists():
                try:
                    with open(yaml_path, 'r') as f:
                        data = yaml.safe_load(f)
                    
                    classes = data.get('names', [])
                    
                    # 重複チェック
                    if class_name in classes:
                        QMessageBox.warning(self, "警告", f"クラス '{class_name}' は既に存在します")
                        return
                    
                    # クラスを追加
                    classes.append(class_name)
                    data['names'] = classes
                    data['nc'] = len(classes)
                    
                    # 保存
                    with open(yaml_path, 'w') as f:
                        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
                    
                    # UI更新
                    self.update_dataset_info()
                    self.notify_auto_save()
                    
                except Exception as e:
                    QMessageBox.critical(self, "エラー", f"クラス追加エラー: {str(e)}")
    
    def remove_class(self):
        """クラスを削除"""
        if not self.dataset_path:
            QMessageBox.warning(self, "警告", "先にデータセットを選択してください")
            return
        
        # 現在のクラスを取得
        yaml_path = self.dataset_path / "data.yaml"
        if not yaml_path.exists():
            return
        
        try:
            with open(yaml_path, 'r') as f:
                data = yaml.safe_load(f)
            
            classes = data.get('names', [])
            
            if not classes:
                QMessageBox.information(self, "情報", "削除するクラスがありません")
                return
            
            # クラス選択ダイアログ
            class_name, ok = QInputDialog.getItem(
                self,
                "クラス削除",
                "削除するクラスを選択してください:",
                classes,
                0,
                False
            )
            
            if ok and class_name:
                # 確認ダイアログ
                reply = QMessageBox.question(
                    self,
                    "確認",
                    f"クラス '{class_name}' を削除しますか？\n注意: このクラスのアノテーションは手動で更新する必要があります。",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                
                if reply == QMessageBox.StandardButton.Yes:
                    # クラスを削除
                    classes.remove(class_name)
                    data['names'] = classes
                    data['nc'] = len(classes)
                    
                    # 保存
                    with open(yaml_path, 'w') as f:
                        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
                    
                    # UI更新
                    self.update_dataset_info()
                    self.notify_auto_save()
                    
        except Exception as e:
            QMessageBox.critical(self, "エラー", f"クラス削除エラー: {str(e)}")