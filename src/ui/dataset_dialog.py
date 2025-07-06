from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
                               QLabel, QLineEdit, QFileDialog, QGroupBox,
                               QRadioButton, QDialogButtonBox, QMessageBox)
from PySide6.QtCore import Qt
from pathlib import Path
import yaml
import shutil

class DatasetSelectionDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("データセット選択")
        self.setModal(True)
        self.resize(600, 400)
        
        self.dataset_path = None
        self.images_path = None
        self.dataset_type = "existing"  # "existing" or "new"
        
        self.init_ui()
    
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
        
        existing_layout = QHBoxLayout()
        self.dataset_path_edit = QLineEdit()
        self.dataset_path_edit.setPlaceholderText("データセットフォルダを選択...")
        self.dataset_path_edit.setReadOnly(True)
        self.browse_dataset_btn = QPushButton("参照...")
        self.browse_dataset_btn.clicked.connect(self.browse_dataset)
        existing_layout.addWidget(self.dataset_path_edit)
        existing_layout.addWidget(self.browse_dataset_btn)
        dataset_layout.addLayout(existing_layout)
        
        # 新規データセット作成
        self.new_radio = QRadioButton("新規データセットを作成")
        self.new_radio.toggled.connect(self.on_dataset_type_changed)
        dataset_layout.addWidget(self.new_radio)
        
        new_layout = QHBoxLayout()
        self.new_dataset_name_edit = QLineEdit()
        self.new_dataset_name_edit.setPlaceholderText("新規データセット名を入力...")
        self.new_dataset_name_edit.setEnabled(False)
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
        
        # 情報表示エリア
        info_group = QGroupBox("データセット情報")
        self.info_label = QLabel("データセットを選択してください")
        self.info_label.setWordWrap(True)
        info_layout = QVBoxLayout()
        info_layout.addWidget(self.info_label)
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)
        
        # ボタン
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | 
            QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept_selection)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
    
    def on_dataset_type_changed(self, checked):
        if self.existing_radio.isChecked():
            self.dataset_type = "existing"
            self.dataset_path_edit.setEnabled(True)
            self.browse_dataset_btn.setEnabled(True)
            self.new_dataset_name_edit.setEnabled(False)
        else:
            self.dataset_type = "new"
            self.dataset_path_edit.setEnabled(False)
            self.browse_dataset_btn.setEnabled(False)
            self.new_dataset_name_edit.setEnabled(True)
    
    def browse_dataset(self):
        folder = QFileDialog.getExistingDirectory(
            self, 
            "データセットフォルダを選択",
            str(Path.home())
        )
        if folder:
            self.dataset_path_edit.setText(folder)
            self.validate_dataset(folder)
    
    def browse_images(self):
        folder = QFileDialog.getExistingDirectory(
            self, 
            "画像フォルダを選択",
            str(Path.home())
        )
        if folder:
            self.images_path_edit.setText(folder)
            self.update_info()
    
    def validate_dataset(self, path):
        """データセットの構造を検証"""
        dataset_path = Path(path)
        
        # data.yamlの存在確認
        yaml_path = dataset_path / "data.yaml"
        if yaml_path.exists():
            try:
                with open(yaml_path, 'r') as f:
                    data = yaml.safe_load(f)
                
                nc = data.get('nc', 0)
                names = data.get('names', [])
                
                info_text = f"データセット: {dataset_path.name}\n"
                info_text += f"クラス数: {nc}\n"
                info_text += f"クラス名: {', '.join(names)}\n"
                
                # train/valフォルダの確認
                train_exists = (dataset_path / "train").exists()
                val_exists = (dataset_path / "valid").exists() or (dataset_path / "val").exists()
                
                if train_exists:
                    train_images = len(list((dataset_path / "train" / "images").glob("*")))
                    info_text += f"訓練画像数: {train_images}\n"
                
                if val_exists:
                    val_dir = "valid" if (dataset_path / "valid").exists() else "val"
                    val_images = len(list((dataset_path / val_dir / "images").glob("*")))
                    info_text += f"検証画像数: {val_images}\n"
                
                self.info_label.setText(info_text)
                self.dataset_path = dataset_path
                
            except Exception as e:
                self.info_label.setText(f"データセット読み込みエラー: {str(e)}")
        else:
            self.info_label.setText("警告: data.yamlが見つかりません")
    
    def update_info(self):
        """情報表示を更新"""
        if self.dataset_type == "new" and self.new_dataset_name_edit.text():
            info_text = f"新規データセット: {self.new_dataset_name_edit.text()}\n"
            info_text += "構造: train/images, train/labels, valid/images, valid/labels\n"
            if self.images_path_edit.text():
                images_path = Path(self.images_path_edit.text())
                image_count = len(list(images_path.glob("*.jpg")) + list(images_path.glob("*.png")))
                info_text += f"アノテーション対象画像数: {image_count}"
            self.info_label.setText(info_text)
    
    def accept_selection(self):
        """選択を確定"""
        # 画像フォルダの確認
        if not self.images_path_edit.text():
            QMessageBox.warning(self, "警告", "画像フォルダを選択してください")
            return
        
        self.images_path = Path(self.images_path_edit.text())
        
        if self.dataset_type == "existing":
            if not self.dataset_path_edit.text():
                QMessageBox.warning(self, "警告", "データセットフォルダを選択してください")
                return
            self.dataset_path = Path(self.dataset_path_edit.text())
        else:
            # 新規データセット作成
            if not self.new_dataset_name_edit.text():
                QMessageBox.warning(self, "警告", "データセット名を入力してください")
                return
            
            # データセットフォルダを作成
            from ..config import BASE_DIR
            self.dataset_path = BASE_DIR / "datasets" / self.new_dataset_name_edit.text()
            
            if self.dataset_path.exists():
                reply = QMessageBox.question(
                    self, 
                    "確認", 
                    f"データセット '{self.new_dataset_name_edit.text()}' は既に存在します。上書きしますか？",
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
            # 初期クラスとして"object"のみを設定
            default_labels = ["object"]
            
            data_yaml = {
                'path': str(self.dataset_path),
                'train': './train/images',
                'val': './valid/images',
                'nc': len(default_labels),
                'names': default_labels
            }
            
            yaml_path = self.dataset_path / 'data.yaml'
            with open(yaml_path, 'w') as f:
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