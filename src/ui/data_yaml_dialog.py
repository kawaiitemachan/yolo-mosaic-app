from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
                               QLabel, QLineEdit, QFileDialog, QDialogButtonBox,
                               QMessageBox, QTextEdit)
from PySide6.QtCore import Qt
from pathlib import Path
import yaml

from ..config import ANNOTATIONS_DIR

class DataYamlDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("データセット設定ファイルの作成")
        self.setModal(True)
        self.resize(600, 400)
        self.yaml_path = None
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        layout.addWidget(QLabel("YOLOトレーニング用のdata.yamlファイルを作成します"))
        
        self.train_layout = self.create_path_selector("訓練データ:")
        layout.addLayout(self.train_layout)
        
        self.val_layout = self.create_path_selector("検証データ:")
        layout.addLayout(self.val_layout)
        
        self.test_layout = self.create_path_selector("テストデータ (オプション):")
        layout.addLayout(self.test_layout)
        
        self.name_layout = QHBoxLayout()
        self.name_layout.addWidget(QLabel("プロジェクト名:"))
        self.name_edit = QLineEdit("my_dataset")
        self.name_layout.addWidget(self.name_edit)
        layout.addLayout(self.name_layout)
        
        preview_btn = QPushButton("プレビュー")
        preview_btn.clicked.connect(self.preview_yaml)
        layout.addWidget(preview_btn)
        
        self.preview_text = QTextEdit()
        self.preview_text.setReadOnly(True)
        layout.addWidget(self.preview_text)
        
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | 
            QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.create_yaml)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
    
    def create_path_selector(self, label):
        layout = QHBoxLayout()
        layout.addWidget(QLabel(label))
        
        line_edit = QLineEdit()
        layout.addWidget(line_edit)
        
        browse_btn = QPushButton("参照")
        browse_btn.clicked.connect(lambda: self.browse_folder(line_edit))
        layout.addWidget(browse_btn)
        
        layout.line_edit = line_edit
        return layout
    
    def browse_folder(self, line_edit):
        folder = QFileDialog.getExistingDirectory(
            self, "フォルダを選択", str(ANNOTATIONS_DIR.parent)
        )
        if folder:
            line_edit.setText(folder)
    
    def get_yaml_data(self):
        from ..config import DEFAULT_CONFIG
        
        train_path = self.train_layout.line_edit.text()
        val_path = self.val_layout.line_edit.text()
        test_path = self.test_layout.line_edit.text()
        
        if not train_path or not val_path:
            return None
        
        labels = DEFAULT_CONFIG["annotation"]["labels"]
        
        data = {
            'path': str(ANNOTATIONS_DIR.parent),
            'train': train_path,
            'val': val_path,
            'nc': len(labels),
            'names': labels
        }
        
        if test_path:
            data['test'] = test_path
        
        return data
    
    def preview_yaml(self):
        data = self.get_yaml_data()
        if data:
            yaml_str = yaml.dump(data, default_flow_style=False, allow_unicode=True)
            self.preview_text.setText(yaml_str)
        else:
            QMessageBox.warning(self, "警告", "訓練データと検証データのパスを指定してください")
    
    def create_yaml(self):
        data = self.get_yaml_data()
        if not data:
            QMessageBox.warning(self, "警告", "訓練データと検証データのパスを指定してください")
            return
        
        name = self.name_edit.text() or "data"
        self.yaml_path = ANNOTATIONS_DIR / f"{name}.yaml"
        
        with open(self.yaml_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
        
        self.accept()