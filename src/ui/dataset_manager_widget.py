from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QTreeWidget,
                               QTreeWidgetItem, QPushButton, QLabel, QTextEdit,
                               QSplitter, QToolBar, QMenu, QMessageBox, QDialog,
                               QDialogButtonBox, QLineEdit, QGroupBox, QGridLayout)
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QAction, QIcon, QFont
from pathlib import Path
import yaml
import shutil
from datetime import datetime
import random

from ..config import DATASETS_DIR

class DatasetInfoDialog(QDialog):
    """データセット情報表示ダイアログ"""
    def __init__(self, dataset_path, parent=None):
        super().__init__(parent)
        self.dataset_path = Path(dataset_path)
        self.setWindowTitle(f"データセット情報: {self.dataset_path.name}")
        self.resize(800, 600)
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # データセット情報
        info_group = QGroupBox("基本情報")
        info_layout = QGridLayout(info_group)
        
        # パス
        info_layout.addWidget(QLabel("パス:"), 0, 0)
        path_label = QLabel(str(self.dataset_path))
        path_label.setWordWrap(True)
        info_layout.addWidget(path_label, 0, 1)
        
        # data.yaml読み込み
        yaml_path = self.dataset_path / "data.yaml"
        if yaml_path.exists():
            try:
                with open(yaml_path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                
                # クラス数
                nc = data.get('nc', 0)
                info_layout.addWidget(QLabel("クラス数:"), 1, 0)
                info_layout.addWidget(QLabel(str(nc)), 1, 1)
                
                # クラス名
                names = data.get('names', [])
                info_layout.addWidget(QLabel("クラス名:"), 2, 0)
                names_text = ', '.join(names) if names else "（未設定）"
                info_layout.addWidget(QLabel(names_text), 2, 1)
                
            except Exception as e:
                info_layout.addWidget(QLabel("エラー:"), 3, 0)
                info_layout.addWidget(QLabel(str(e)), 3, 1)
        
        layout.addWidget(info_group)
        
        # データ分布
        dist_group = QGroupBox("データ分布")
        dist_layout = QVBoxLayout(dist_group)
        
        dist_text = QTextEdit()
        dist_text.setReadOnly(True)
        dist_text.setFont(QFont("Monaco", 10))
        
        # データ分布を計算
        dist_info = self.calculate_distribution()
        dist_text.setPlainText(dist_info)
        
        dist_layout.addWidget(dist_text)
        layout.addWidget(dist_group)
        
        # サンプル画像プレビュー
        preview_group = QGroupBox("サンプル画像")
        preview_layout = QVBoxLayout(preview_group)
        
        self.preview_label = QLabel("サンプル画像がここに表示されます")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumHeight(200)
        self.preview_label.setStyleSheet("QLabel { border: 1px solid #ccc; background: #f0f0f0; }")
        preview_layout.addWidget(self.preview_label)
        
        # プレビューボタン
        preview_btn = QPushButton("ランダムに画像を表示")
        preview_btn.clicked.connect(self.show_random_image)
        preview_layout.addWidget(preview_btn)
        
        layout.addWidget(preview_group)
        
        # ボタン
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)
        
        # 初回プレビュー
        self.show_random_image()
    
    def calculate_distribution(self):
        """データ分布を計算"""
        info = []
        
        # train
        train_path = self.dataset_path / "train"
        if train_path.exists():
            train_images = len(list((train_path / "images").glob("*.*"))) if (train_path / "images").exists() else 0
            train_labels = len(list((train_path / "labels").glob("*.txt"))) if (train_path / "labels").exists() else 0
            info.append(f"訓練データ:")
            info.append(f"  画像: {train_images} 枚")
            info.append(f"  ラベル: {train_labels} 件")
            info.append("")
        
        # valid/val
        for val_name in ["valid", "val"]:
            val_path = self.dataset_path / val_name
            if val_path.exists():
                val_images = len(list((val_path / "images").glob("*.*"))) if (val_path / "images").exists() else 0
                val_labels = len(list((val_path / "labels").glob("*.txt"))) if (val_path / "labels").exists() else 0
                info.append(f"検証データ ({val_name}):")
                info.append(f"  画像: {val_images} 枚")
                info.append(f"  ラベル: {val_labels} 件")
                info.append("")
                break
        
        # 合計
        total_images = 0
        total_labels = 0
        for split_dir in self.dataset_path.iterdir():
            if split_dir.is_dir() and split_dir.name in ["train", "valid", "val"]:
                images_dir = split_dir / "images"
                labels_dir = split_dir / "labels"
                if images_dir.exists():
                    total_images += len(list(images_dir.glob("*.*")))
                if labels_dir.exists():
                    total_labels += len(list(labels_dir.glob("*.txt")))
        
        info.append("合計:")
        info.append(f"  画像: {total_images} 枚")
        info.append(f"  ラベル: {total_labels} 件")
        
        # ラベル率
        if total_images > 0:
            label_rate = (total_labels / total_images) * 100
            info.append(f"  ラベル率: {label_rate:.1f}%")
        
        # 最終更新日時
        if self.dataset_path.exists():
            mtime = datetime.fromtimestamp(self.dataset_path.stat().st_mtime)
            info.append(f"\n最終更新: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
        
        return "\n".join(info)
    
    def show_random_image(self):
        """ランダムに画像を表示"""
        # 全画像を収集
        all_images = []
        for split_dir in self.dataset_path.iterdir():
            if split_dir.is_dir() and split_dir.name in ["train", "valid", "val"]:
                images_dir = split_dir / "images"
                if images_dir.exists():
                    all_images.extend(list(images_dir.glob("*.jpg")) + 
                                    list(images_dir.glob("*.jpeg")) + 
                                    list(images_dir.glob("*.png")))
        
        if all_images:
            # ランダムに選択
            selected_image = random.choice(all_images)
            
            # QPixmapで表示
            from PySide6.QtGui import QPixmap
            pixmap = QPixmap(str(selected_image))
            if not pixmap.isNull():
                # サイズ調整
                scaled_pixmap = pixmap.scaled(
                    400, 300,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
                self.preview_label.setPixmap(scaled_pixmap)
                self.preview_label.setText("")
            else:
                self.preview_label.setText("画像を読み込めませんでした")
        else:
            self.preview_label.setText("画像が見つかりません")

class DatasetManagerWidget(QWidget):
    """データセット管理ウィジェット"""
    dataset_selected = Signal(str)  # データセットが選択されたときのシグナル
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.load_datasets()
        
        # 自動更新タイマー
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.load_datasets)
        self.refresh_timer.start(5000)  # 5秒ごとに更新
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # ツールバー
        toolbar = QToolBar()
        
        refresh_action = QAction("更新", self)
        refresh_action.triggered.connect(self.load_datasets)
        toolbar.addAction(refresh_action)
        
        toolbar.addSeparator()
        
        info_action = QAction("詳細情報", self)
        info_action.triggered.connect(self.show_info)
        toolbar.addAction(info_action)
        
        rename_action = QAction("名前変更", self)
        rename_action.triggered.connect(self.rename_dataset)
        toolbar.addAction(rename_action)
        
        delete_action = QAction("削除", self)
        delete_action.triggered.connect(self.delete_dataset)
        toolbar.addAction(delete_action)
        
        toolbar.addSeparator()
        
        open_action = QAction("Finderで開く", self)
        open_action.triggered.connect(self.open_in_finder)
        toolbar.addAction(open_action)
        
        layout.addWidget(toolbar)
        
        # スプリッター
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # 左側：データセット一覧
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["データセット", "クラス数", "画像数", "最終更新"])
        self.tree.itemSelectionChanged.connect(self.on_selection_changed)
        self.tree.itemDoubleClicked.connect(self.show_info)
        
        # コンテキストメニュー
        self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self.show_context_menu)
        
        left_layout.addWidget(self.tree)
        
        # 右側：詳細情報
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        self.info_label = QLabel("データセット情報")
        self.info_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        right_layout.addWidget(self.info_label)
        
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        right_layout.addWidget(self.info_text)
        
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 1)
        
        layout.addWidget(splitter)
    
    def load_datasets(self):
        """データセット一覧を読み込み"""
        self.tree.clear()
        
        if not DATASETS_DIR.exists():
            return
        
        for dataset_dir in sorted(DATASETS_DIR.iterdir()):
            if dataset_dir.is_dir():
                # data.yamlの存在確認
                yaml_path = dataset_dir / "data.yaml"
                if yaml_path.exists():
                    item = QTreeWidgetItem()
                    item.setText(0, dataset_dir.name)
                    item.setData(0, Qt.ItemDataRole.UserRole, str(dataset_dir))
                    
                    # data.yaml読み込み
                    try:
                        with open(yaml_path, 'r', encoding='utf-8') as f:
                            data = yaml.safe_load(f)
                        
                        nc = data.get('nc', 0)
                        item.setText(1, str(nc))
                        
                        # 画像数をカウント
                        total_images = 0
                        for split_dir in dataset_dir.iterdir():
                            if split_dir.is_dir() and split_dir.name in ["train", "valid", "val"]:
                                images_dir = split_dir / "images"
                                if images_dir.exists():
                                    total_images += len(list(images_dir.glob("*.*")))
                        
                        item.setText(2, str(total_images))
                        
                    except Exception:
                        item.setText(1, "エラー")
                        item.setText(2, "-")
                    
                    # 最終更新日時
                    mtime = datetime.fromtimestamp(dataset_dir.stat().st_mtime)
                    item.setText(3, mtime.strftime("%Y-%m-%d %H:%M"))
                    
                    self.tree.addTopLevelItem(item)
    
    def on_selection_changed(self):
        """選択が変更されたときの処理"""
        current = self.tree.currentItem()
        if current:
            dataset_path = Path(current.data(0, Qt.ItemDataRole.UserRole))
            self.display_dataset_info(dataset_path)
            self.dataset_selected.emit(str(dataset_path))
    
    def display_dataset_info(self, dataset_path):
        """データセット情報を表示"""
        self.info_label.setText(f"データセット: {dataset_path.name}")
        
        info_text = []
        info_text.append(f"パス: {dataset_path}")
        info_text.append("")
        
        # data.yaml読み込み
        yaml_path = dataset_path / "data.yaml"
        if yaml_path.exists():
            try:
                with open(yaml_path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                
                nc = data.get('nc', 0)
                names = data.get('names', [])
                
                info_text.append(f"クラス数: {nc}")
                if names:
                    info_text.append(f"クラス名: {', '.join(names)}")
                else:
                    info_text.append("クラス名: （未設定）")
                
            except Exception as e:
                info_text.append(f"data.yaml読み込みエラー: {str(e)}")
        
        info_text.append("")
        info_text.append("【データ分布】")
        
        # train
        train_path = dataset_path / "train"
        if train_path.exists():
            train_images = len(list((train_path / "images").glob("*.*"))) if (train_path / "images").exists() else 0
            train_labels = len(list((train_path / "labels").glob("*.txt"))) if (train_path / "labels").exists() else 0
            info_text.append(f"訓練データ: 画像 {train_images} 枚, ラベル {train_labels} 件")
        
        # valid/val
        for val_name in ["valid", "val"]:
            val_path = dataset_path / val_name
            if val_path.exists():
                val_images = len(list((val_path / "images").glob("*.*"))) if (val_path / "images").exists() else 0
                val_labels = len(list((val_path / "labels").glob("*.txt"))) if (val_path / "labels").exists() else 0
                info_text.append(f"検証データ: 画像 {val_images} 枚, ラベル {val_labels} 件")
                break
        
        self.info_text.setPlainText("\n".join(info_text))
    
    def show_context_menu(self, position):
        """コンテキストメニューを表示"""
        item = self.tree.itemAt(position)
        if not item:
            return
        
        menu = QMenu(self)
        
        info_action = menu.addAction("詳細情報")
        info_action.triggered.connect(self.show_info)
        
        menu.addSeparator()
        
        rename_action = menu.addAction("名前変更")
        rename_action.triggered.connect(self.rename_dataset)
        
        delete_action = menu.addAction("削除")
        delete_action.triggered.connect(self.delete_dataset)
        
        menu.addSeparator()
        
        open_action = menu.addAction("Finderで開く")
        open_action.triggered.connect(self.open_in_finder)
        
        menu.exec(self.tree.mapToGlobal(position))
    
    def show_info(self):
        """詳細情報ダイアログを表示"""
        current = self.tree.currentItem()
        if current:
            dataset_path = current.data(0, Qt.ItemDataRole.UserRole)
            dialog = DatasetInfoDialog(dataset_path, self)
            dialog.exec()
    
    def rename_dataset(self):
        """データセット名を変更"""
        current = self.tree.currentItem()
        if not current:
            return
        
        dataset_path = Path(current.data(0, Qt.ItemDataRole.UserRole))
        old_name = dataset_path.name
        
        # 入力ダイアログ
        dialog = QDialog(self)
        dialog.setWindowTitle("データセット名の変更")
        dialog.resize(400, 120)
        
        layout = QVBoxLayout(dialog)
        
        layout.addWidget(QLabel(f"現在の名前: {old_name}"))
        
        name_edit = QLineEdit(old_name)
        name_edit.selectAll()
        layout.addWidget(name_edit)
        
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | 
            QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            new_name = name_edit.text().strip()
            if new_name and new_name != old_name:
                new_path = dataset_path.parent / new_name
                
                if new_path.exists():
                    QMessageBox.warning(
                        self, 
                        "エラー", 
                        f"'{new_name}' という名前のデータセットは既に存在します"
                    )
                    return
                
                try:
                    dataset_path.rename(new_path)
                    
                    # data.yamlのpathを更新
                    yaml_path = new_path / "data.yaml"
                    if yaml_path.exists():
                        with open(yaml_path, 'r', encoding='utf-8') as f:
                            data = yaml.safe_load(f)
                        
                        data['path'] = str(new_path)
                        
                        with open(yaml_path, 'w', encoding='utf-8') as f:
                            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
                    
                    self.load_datasets()
                    QMessageBox.information(
                        self, 
                        "成功", 
                        f"データセット名を '{new_name}' に変更しました"
                    )
                except Exception as e:
                    QMessageBox.critical(
                        self, 
                        "エラー", 
                        f"名前の変更に失敗しました: {str(e)}"
                    )
    
    def delete_dataset(self):
        """データセットを削除"""
        current = self.tree.currentItem()
        if not current:
            return
        
        dataset_path = Path(current.data(0, Qt.ItemDataRole.UserRole))
        dataset_name = dataset_path.name
        
        reply = QMessageBox.question(
            self,
            "確認",
            f"データセット '{dataset_name}' を削除しますか？\n"
            "この操作は取り消せません。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            try:
                shutil.rmtree(dataset_path)
                self.load_datasets()
                self.info_text.clear()
                self.info_label.setText("データセット情報")
                QMessageBox.information(
                    self, 
                    "成功", 
                    f"データセット '{dataset_name}' を削除しました"
                )
            except Exception as e:
                QMessageBox.critical(
                    self, 
                    "エラー", 
                    f"削除に失敗しました: {str(e)}"
                )
    
    def open_in_finder(self):
        """Finderでデータセットフォルダを開く"""
        current = self.tree.currentItem()
        if current:
            dataset_path = current.data(0, Qt.ItemDataRole.UserRole)
            import subprocess
            subprocess.run(["open", dataset_path])