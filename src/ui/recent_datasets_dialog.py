from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, 
                               QListWidget, QListWidgetItem, QPushButton,
                               QLabel, QDialogButtonBox, QMessageBox)
from PySide6.QtCore import Qt
from pathlib import Path
from datetime import datetime

class RecentDatasetsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("最近使用したデータセット")
        self.setModal(True)
        self.resize(600, 400)
        
        self.selected_dataset = None
        self.init_ui()
        self.load_recent_datasets()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # 説明ラベル
        info_label = QLabel("最近使用したデータセットを選択してください：")
        layout.addWidget(info_label)
        
        # データセットリスト
        self.dataset_list = QListWidget()
        self.dataset_list.itemDoubleClicked.connect(self.on_item_double_clicked)
        layout.addWidget(self.dataset_list)
        
        # ボタン
        button_layout = QHBoxLayout()
        
        self.open_btn = QPushButton("開く")
        self.open_btn.clicked.connect(self.accept_selection)
        self.open_btn.setEnabled(False)
        
        self.remove_btn = QPushButton("リストから削除")
        self.remove_btn.clicked.connect(self.remove_dataset)
        self.remove_btn.setEnabled(False)
        
        button_layout.addWidget(self.open_btn)
        button_layout.addWidget(self.remove_btn)
        button_layout.addStretch()
        
        layout.addLayout(button_layout)
        
        # ダイアログボタン
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        
        # 選択変更時の処理
        self.dataset_list.currentItemChanged.connect(self.on_selection_changed)
    
    def load_recent_datasets(self):
        """最近使用したデータセットを読み込む"""
        from ..utils.settings_manager import SettingsManager
        self.settings_manager = SettingsManager()
        
        recent_datasets = self.settings_manager.get_recent_datasets()
        
        for dataset_info in recent_datasets:
            path = Path(dataset_info["path"])
            
            # リストアイテムを作成
            item = QListWidgetItem()
            
            # 表示テキスト
            display_text = f"{dataset_info['name']}\n"
            display_text += f"パス: {path}\n"
            
            # 最終使用日時
            try:
                last_used = datetime.fromisoformat(dataset_info["last_used"])
                display_text += f"最終使用: {last_used.strftime('%Y-%m-%d %H:%M')}"
            except:
                pass
            
            item.setText(display_text)
            item.setData(Qt.ItemDataRole.UserRole, str(path))
            
            # データセットの検証
            if self.validate_dataset(path):
                item.setToolTip("クリックして選択、ダブルクリックで開く")
            else:
                item.setDisabled(True)
                item.setToolTip("データセットが見つかりません")
            
            self.dataset_list.addItem(item)
    
    def validate_dataset(self, path):
        """データセットの妥当性を確認"""
        dataset_path = Path(path)
        return dataset_path.exists() and (dataset_path / "data.yaml").exists()
    
    def on_selection_changed(self, current, previous):
        """選択が変更されたときの処理"""
        if current and not current.isDisabled():
            self.open_btn.setEnabled(True)
            self.remove_btn.setEnabled(True)
        else:
            self.open_btn.setEnabled(False)
            self.remove_btn.setEnabled(False)
    
    def on_item_double_clicked(self, item):
        """アイテムがダブルクリックされたときの処理"""
        if not item.isDisabled():
            self.accept_selection()
    
    def accept_selection(self):
        """選択を確定"""
        current_item = self.dataset_list.currentItem()
        if current_item and not current_item.isDisabled():
            self.selected_dataset = Path(current_item.data(Qt.ItemDataRole.UserRole))
            self.accept()
    
    def remove_dataset(self):
        """選択されたデータセットをリストから削除"""
        current_item = self.dataset_list.currentItem()
        if not current_item:
            return
        
        reply = QMessageBox.question(
            self,
            "確認",
            "選択されたデータセットを最近使用したリストから削除しますか？\n（データセット自体は削除されません）",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            dataset_path = current_item.data(Qt.ItemDataRole.UserRole)
            
            # リストから削除
            self.settings_manager.recent_datasets = [
                d for d in self.settings_manager.recent_datasets
                if d["path"] != dataset_path
            ]
            self.settings_manager.save_recent_datasets()
            
            # UIから削除
            row = self.dataset_list.row(current_item)
            self.dataset_list.takeItem(row)