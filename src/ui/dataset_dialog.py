from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
                               QLabel, QLineEdit, QFileDialog, QGroupBox,
                               QRadioButton, QDialogButtonBox, QMessageBox,
                               QListWidget, QListWidgetItem, QSplitter,
                               QWidget, QTextEdit)
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon, QFont
from pathlib import Path
import yaml
import shutil
from datetime import datetime

class DatasetListWidget(QWidget):
    """データセット一覧表示ウィジェット"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.selected_dataset_path = None
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # データセットリスト
        self.dataset_list = QListWidget()
        self.dataset_list.itemSelectionChanged.connect(self.on_selection_changed)
        self.dataset_list.setStyleSheet("""
            QListWidget {
                border: 1px solid #ddd;
                border-radius: 5px;
                padding: 5px;
                font-size: 14px;
            }
            QListWidget::item {
                padding: 8px;
                margin: 2px;
                border-radius: 3px;
            }
            QListWidget::item:selected {
                background-color: #3b82f6;
                color: white;
            }
            QListWidget::item:hover {
                background-color: #e0e7ff;
            }
        """)
        layout.addWidget(self.dataset_list)
        
        # 更新ボタン
        refresh_btn = QPushButton("リストを更新")
        refresh_btn.clicked.connect(self.load_datasets)
        layout.addWidget(refresh_btn)
        
    def load_datasets(self):
        """datasetsフォルダからデータセットを読み込む"""
        self.dataset_list.clear()
        
        # datasetsフォルダのパスを取得
        from ..config import DATASETS_DIR
        
        if not DATASETS_DIR.exists():
            DATASETS_DIR.mkdir(parents=True, exist_ok=True)
            return
            
        # データセットを検索
        for item in sorted(DATASETS_DIR.iterdir()):
            if item.is_dir():
                # data.yamlの存在を確認
                yaml_path = item / "data.yaml"
                if yaml_path.exists():
                    # リストアイテムを作成
                    list_item = QListWidgetItem(item.name)
                    list_item.setData(Qt.ItemDataRole.UserRole, str(item))
                    
                    # 有効なデータセットかどうかチェック
                    if self.is_valid_dataset(item):
                        list_item.setIcon(QIcon())  # 有効なアイコン（今は空）
                    else:
                        list_item.setForeground(Qt.GlobalColor.gray)
                        list_item.setToolTip("データセット構造が不完全です")
                    
                    self.dataset_list.addItem(list_item)
    
    def is_valid_dataset(self, dataset_path):
        """データセットの妥当性を確認"""
        # 必要なディレクトリの存在を確認
        required_dirs = [
            dataset_path / "train" / "images",
            dataset_path / "train" / "labels"
        ]
        
        # valid または val ディレクトリも確認
        valid_dirs = [
            dataset_path / "valid" / "images",
            dataset_path / "valid" / "labels"
        ]
        val_dirs = [
            dataset_path / "val" / "images",
            dataset_path / "val" / "labels"
        ]
        
        # trainディレクトリは必須
        for dir_path in required_dirs:
            if not dir_path.exists():
                return False
                
        # validまたはvalディレクトリのどちらかが存在すればOK
        valid_exists = all(d.exists() for d in valid_dirs)
        val_exists = all(d.exists() for d in val_dirs)
        
        return valid_exists or val_exists
    
    def on_selection_changed(self):
        """選択が変更されたときの処理"""
        current_item = self.dataset_list.currentItem()
        if current_item:
            self.selected_dataset_path = Path(current_item.data(Qt.ItemDataRole.UserRole))
            # 親ダイアログに通知
            parent = self.parent()
            while parent and not isinstance(parent, DatasetSelectionDialog):
                parent = parent.parent()
            if parent:
                parent.on_dataset_selected(self.selected_dataset_path)
        else:
            self.selected_dataset_path = None

class DatasetSelectionDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("データセット選択")
        self.setModal(True)
        self.resize(800, 600)
        
        self.dataset_path = None
        self.images_path = None
        self.dataset_type = "existing"  # "existing" or "new"
        
        self.init_ui()
        
        # 初期データセットの読み込み
        self.dataset_list_widget.load_datasets()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # データセット選択セクション
        dataset_group = QGroupBox("データセット選択")
        dataset_layout = QVBoxLayout()
        
        # 既存データセット選択
        self.existing_radio = QRadioButton("既存のデータセットを使用")
        self.existing_radio.setChecked(True)
        self.existing_radio.toggled.connect(self.on_dataset_type_changed)
        dataset_layout.addWidget(self.existing_radio)
        
        # スプリッターで左右に分割
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # 左側：データセットリスト
        self.dataset_list_widget = DatasetListWidget()
        splitter.addWidget(self.dataset_list_widget)
        
        # 右側：詳細情報
        self.info_widget = QWidget()
        info_layout = QVBoxLayout(self.info_widget)
        
        info_label = QLabel("データセット詳細")
        info_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        info_layout.addWidget(info_label)
        
        self.dataset_info_text = QTextEdit()
        self.dataset_info_text.setReadOnly(True)
        self.dataset_info_text.setPlainText("データセットを選択してください")
        self.dataset_info_text.setStyleSheet("""
            QTextEdit {
                border: 1px solid #ddd;
                border-radius: 5px;
                padding: 10px;
                background-color: #f9fafb;
                font-family: 'Monaco', 'Menlo', 'Consolas', monospace;
                font-size: 13px;
            }
        """)
        info_layout.addWidget(self.dataset_info_text)
        
        splitter.addWidget(self.info_widget)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        
        dataset_layout.addWidget(splitter)
        
        # 新規データセット作成
        self.new_radio = QRadioButton("新規データセットを作成")
        self.new_radio.toggled.connect(self.on_dataset_type_changed)
        dataset_layout.addWidget(self.new_radio)
        
        new_layout = QHBoxLayout()
        self.new_dataset_name_edit = QLineEdit()
        self.new_dataset_name_edit.setPlaceholderText("新規データセット名を入力...")
        self.new_dataset_name_edit.setEnabled(False)
        self.new_dataset_name_edit.textChanged.connect(self.update_new_dataset_info)
        new_layout.addWidget(QLabel("データセット名:"))
        new_layout.addWidget(self.new_dataset_name_edit)
        dataset_layout.addLayout(new_layout)
        
        dataset_group.setLayout(dataset_layout)
        layout.addWidget(dataset_group)
        
        # 画像フォルダ選択セクション
        images_group = QGroupBox("アノテーションする画像フォルダ")
        images_layout = QHBoxLayout()
        
        self.images_path_edit = QLineEdit()
        self.images_path_edit.setPlaceholderText("画像フォルダを選択...")
        self.images_path_edit.setReadOnly(True)
        self.browse_images_btn = QPushButton("参照...")
        self.browse_images_btn.clicked.connect(self.browse_images)
        
        images_layout.addWidget(self.images_path_edit)
        images_layout.addWidget(self.browse_images_btn)
        images_group.setLayout(images_layout)
        layout.addWidget(images_group)
        
        # ボタン
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | 
            QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept_selection)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
    
    def on_dataset_type_changed(self, checked):
        """データセットタイプが変更されたときの処理"""
        if self.existing_radio.isChecked():
            self.dataset_type = "existing"
            self.dataset_list_widget.setEnabled(True)
            self.info_widget.setEnabled(True)
            self.new_dataset_name_edit.setEnabled(False)
            
            # 既存の選択を再度表示
            self.dataset_list_widget.on_selection_changed()
        else:
            self.dataset_type = "new"
            self.dataset_list_widget.setEnabled(False)
            self.info_widget.setEnabled(False)
            self.new_dataset_name_edit.setEnabled(True)
            
            # 新規データセット情報を表示
            self.update_new_dataset_info()
    
    def on_dataset_selected(self, dataset_path):
        """データセットが選択されたときの処理"""
        if self.dataset_type == "existing":
            self.dataset_path = dataset_path
            self.display_dataset_info(dataset_path)
    
    def display_dataset_info(self, dataset_path):
        """データセットの詳細情報を表示"""
        info_text = f"データセット: {dataset_path.name}\n"
        info_text += f"パス: {dataset_path}\n\n"
        
        # data.yamlの読み込み
        yaml_path = dataset_path / "data.yaml"
        if yaml_path.exists():
            try:
                with open(yaml_path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                
                nc = data.get('nc', 0)
                names = data.get('names', [])
                
                info_text += f"クラス数: {nc}\n"
                if names:
                    info_text += f"クラス名: {', '.join(names)}\n"
                else:
                    info_text += "クラス名: なし（アノテーション時に追加）\n"
                
                # 画像数の確認
                info_text += "\n【データ分布】\n"
                
                # train
                train_path = dataset_path / "train"
                if train_path.exists():
                    train_images = len(list((train_path / "images").glob("*.*"))) if (train_path / "images").exists() else 0
                    train_labels = len(list((train_path / "labels").glob("*.txt"))) if (train_path / "labels").exists() else 0
                    info_text += f"訓練データ: 画像 {train_images} 枚, ラベル {train_labels} 件\n"
                
                # valid/val
                for val_name in ["valid", "val"]:
                    val_path = dataset_path / val_name
                    if val_path.exists():
                        val_images = len(list((val_path / "images").glob("*.*"))) if (val_path / "images").exists() else 0
                        val_labels = len(list((val_path / "labels").glob("*.txt"))) if (val_path / "labels").exists() else 0
                        info_text += f"検証データ: 画像 {val_images} 枚, ラベル {val_labels} 件\n"
                        break
                
                # 最終更新日時
                if dataset_path.exists():
                    mtime = datetime.fromtimestamp(dataset_path.stat().st_mtime)
                    info_text += f"\n最終更新: {mtime.strftime('%Y-%m-%d %H:%M:%S')}"
                
            except Exception as e:
                info_text += f"\nエラー: {str(e)}"
        else:
            info_text += "\n警告: data.yamlが見つかりません"
        
        self.dataset_info_text.setPlainText(info_text)
    
    def update_new_dataset_info(self):
        """新規データセット作成時の情報を更新"""
        if self.dataset_type == "new" and self.new_dataset_name_edit.text():
            info_text = f"新規データセット: {self.new_dataset_name_edit.text()}\n\n"
            info_text += "【作成される構造】\n"
            info_text += f"datasets/{self.new_dataset_name_edit.text()}/\n"
            info_text += "├── train/\n"
            info_text += "│   ├── images/    ← 訓練用画像\n"
            info_text += "│   └── labels/    ← 訓練用ラベル\n"
            info_text += "├── valid/\n"
            info_text += "│   ├── images/    ← 検証用画像\n"
            info_text += "│   └── labels/    ← 検証用ラベル\n"
            info_text += "└── data.yaml      ← データセット設定\n\n"
            
            info_text += "注意: クラスはアノテーション開始時に追加してください"
            
            self.dataset_info_text.setPlainText(info_text)
        elif self.dataset_type == "new":
            self.dataset_info_text.setPlainText("新規データセット名を入力してください")
    
    def browse_images(self):
        folder = QFileDialog.getExistingDirectory(
            self, 
            "画像フォルダを選択",
            str(Path.home())
        )
        if folder:
            self.images_path_edit.setText(folder)
            # 画像数を表示
            images_path = Path(folder)
            image_count = len(list(images_path.glob("*.jpg")) + list(images_path.glob("*.png")))
            if image_count > 0:
                self.images_path_edit.setToolTip(f"{image_count} 枚の画像が見つかりました")
            else:
                self.images_path_edit.setToolTip("画像が見つかりませんでした")
    
    def accept_selection(self):
        """選択を確定"""
        # 画像フォルダの確認
        if not self.images_path_edit.text():
            QMessageBox.warning(self, "警告", "画像フォルダを選択してください")
            return
        
        self.images_path = Path(self.images_path_edit.text())
        
        if self.dataset_type == "existing":
            if not self.dataset_path:
                QMessageBox.warning(self, "警告", "データセットを選択してください")
                return
        else:
            # 新規データセット作成
            if not self.new_dataset_name_edit.text():
                QMessageBox.warning(self, "警告", "データセット名を入力してください")
                return
            
            # データセット名の妥当性チェック
            dataset_name = self.new_dataset_name_edit.text().strip()
            invalid_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|', ' ']
            if any(char in dataset_name for char in invalid_chars):
                QMessageBox.warning(
                    self, 
                    "警告", 
                    "データセット名に使用できない文字が含まれています:\n/ \\ : * ? \" < > | スペース"
                )
                return
            
            # データセットフォルダを作成
            from ..config import DATASETS_DIR
            self.dataset_path = DATASETS_DIR / dataset_name
            
            if self.dataset_path.exists():
                reply = QMessageBox.question(
                    self, 
                    "確認", 
                    f"データセット '{dataset_name}' は既に存在します。上書きしますか？",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                if reply == QMessageBox.StandardButton.No:
                    return
            
            self.create_new_dataset()
        
        self.accept()
    
    def create_new_dataset(self):
        """新規データセットの構造を作成"""
        try:
            # ディレクトリ構造を作成
            self.dataset_path.mkdir(parents=True, exist_ok=True)
            
            # train/valid構造を作成
            for split in ['train', 'valid']:
                (self.dataset_path / split / 'images').mkdir(parents=True, exist_ok=True)
                (self.dataset_path / split / 'labels').mkdir(parents=True, exist_ok=True)
            
            # デフォルトのdata.yamlを作成
            # 初期クラスは空にする（ユーザーに設定させる）
            data_yaml = {
                'path': str(self.dataset_path),
                'train': './train/images',
                'val': './valid/images',
                'nc': 0,
                'names': []
            }
            
            yaml_path = self.dataset_path / 'data.yaml'
            with open(yaml_path, 'w', encoding='utf-8') as f:
                yaml.dump(data_yaml, f, default_flow_style=False, allow_unicode=True)
            
            QMessageBox.information(
                self, 
                "成功", 
                f"データセット '{self.new_dataset_name_edit.text()}' を作成しました"
            )
            
        except Exception as e:
            QMessageBox.critical(
                self, 
                "エラー", 
                f"データセット作成中にエラーが発生しました: {str(e)}"
            )
            raise