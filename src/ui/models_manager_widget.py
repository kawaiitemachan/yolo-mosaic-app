from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                               QTreeWidget, QTreeWidgetItem, QLabel, QMessageBox,
                               QInputDialog, QMenu, QHeaderView, QToolBar,
                               QDialog, QDialogButtonBox, QTextEdit, QGroupBox)
from PySide6.QtCore import Qt, Signal, QDateTime, QTimer
from PySide6.QtGui import QAction, QIcon, QFont
from pathlib import Path
import shutil
import yaml
import json
from datetime import datetime

from ..config import MODELS_DIR

class ModelInfoDialog(QDialog):
    """モデル情報を表示するダイアログ"""
    def __init__(self, model_path, parent=None):
        super().__init__(parent)
        self.model_path = Path(model_path)
        self.setWindowTitle(f"モデル情報: {self.model_path.name}")
        self.setModal(True)
        self.resize(600, 500)
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # 基本情報グループ
        basic_group = QGroupBox("基本情報")
        basic_layout = QVBoxLayout(basic_group)
        
        # フォルダ名
        name_label = QLabel(f"<b>モデル名:</b> {self.model_path.name}")
        basic_layout.addWidget(name_label)
        
        # 作成日時
        if self.model_path.exists():
            created_time = datetime.fromtimestamp(self.model_path.stat().st_ctime)
            created_label = QLabel(f"<b>作成日時:</b> {created_time.strftime('%Y-%m-%d %H:%M:%S')}")
            basic_layout.addWidget(created_label)
            
            # 最終更新日時
            modified_time = datetime.fromtimestamp(self.model_path.stat().st_mtime)
            modified_label = QLabel(f"<b>最終更新:</b> {modified_time.strftime('%Y-%m-%d %H:%M:%S')}")
            basic_layout.addWidget(modified_label)
        
        layout.addWidget(basic_group)
        
        # 学習設定グループ
        args_path = self.model_path / "args.yaml"
        if args_path.exists():
            args_group = QGroupBox("学習設定")
            args_layout = QVBoxLayout(args_group)
            
            args_text = QTextEdit()
            args_text.setReadOnly(True)
            args_text.setMaximumHeight(200)
            
            try:
                with open(args_path, 'r', encoding='utf-8') as f:
                    args_data = yaml.safe_load(f)
                    args_text.setPlainText(yaml.dump(args_data, allow_unicode=True, sort_keys=False))
            except Exception as e:
                args_text.setPlainText(f"読み込みエラー: {str(e)}")
            
            args_layout.addWidget(args_text)
            layout.addWidget(args_group)
        
        # 学習結果グループ
        results_path = self.model_path / "results.csv"
        if results_path.exists():
            results_group = QGroupBox("学習結果")
            results_layout = QVBoxLayout(results_group)
            
            # 最終エポックの情報を表示
            try:
                with open(results_path, 'r') as f:
                    lines = f.readlines()
                    if len(lines) > 1:
                        last_line = lines[-1].strip()
                        headers = lines[0].strip().split(',')
                        values = last_line.split(',')
                        
                        if len(headers) == len(values):
                            # 主要な指標を表示
                            metrics_text = ""
                            important_metrics = [
                                'epoch', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)',
                                'train/box_loss', 'train/seg_loss', 'train/cls_loss'
                            ]
                            
                            for metric in important_metrics:
                                if metric in headers:
                                    idx = headers.index(metric)
                                    value = values[idx]
                                    try:
                                        value = f"{float(value):.4f}"
                                    except:
                                        pass
                                    metrics_text += f"<b>{metric}:</b> {value}\n"
                            
                            metrics_label = QLabel(metrics_text.strip())
                            results_layout.addWidget(metrics_label)
            except Exception as e:
                error_label = QLabel(f"結果読み込みエラー: {str(e)}")
                results_layout.addWidget(error_label)
            
            layout.addWidget(results_group)
        
        # ボタン
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        button_box.accepted.connect(self.accept)
        layout.addWidget(button_box)

class ModelsManagerWidget(QWidget):
    """モデル管理ウィジェット"""
    model_selected = Signal(str)  # モデルが選択されたときのシグナル
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.load_models()
        
        # 定期的に更新
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.refresh_models)
        self.update_timer.start(5000)  # 5秒ごとに更新
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # ツールバー
        toolbar = QToolBar()
        toolbar.setMovable(False)
        
        # 更新ボタン
        refresh_action = QAction("更新", self)
        refresh_action.triggered.connect(self.refresh_models)
        toolbar.addAction(refresh_action)
        
        # 削除ボタン
        delete_action = QAction("削除", self)
        delete_action.triggered.connect(self.delete_model)
        toolbar.addAction(delete_action)
        
        # 名前変更ボタン
        rename_action = QAction("名前変更", self)
        rename_action.triggered.connect(self.rename_model)
        toolbar.addAction(rename_action)
        
        # 情報表示ボタン
        info_action = QAction("詳細情報", self)
        info_action.triggered.connect(self.show_model_info)
        toolbar.addAction(info_action)
        
        toolbar.addSeparator()
        
        # フォルダを開くボタン
        open_folder_action = QAction("Finderで開く", self)
        open_folder_action.triggered.connect(self.open_in_finder)
        toolbar.addAction(open_folder_action)
        
        layout.addWidget(toolbar)
        
        # 説明ラベル
        info_label = QLabel("学習済みモデルの一覧です。右クリックでメニューを表示します。")
        info_label.setWordWrap(True)
        info_label.setStyleSheet("QLabel { padding: 10px; background-color: #f0f0f0; border-radius: 5px; }")
        layout.addWidget(info_label)
        
        # モデルリスト
        self.model_tree = QTreeWidget()
        self.model_tree.setHeaderLabels(["モデル名", "作成日時", "サイズ", "状態"])
        self.model_tree.setRootIsDecorated(False)
        self.model_tree.setSortingEnabled(True)
        self.model_tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.model_tree.customContextMenuRequested.connect(self.show_context_menu)
        self.model_tree.itemDoubleClicked.connect(self.on_item_double_clicked)
        self.model_tree.itemSelectionChanged.connect(self.on_selection_changed)
        
        # ヘッダーの調整
        header = self.model_tree.header()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        
        layout.addWidget(self.model_tree)
        
        # ステータスラベル
        self.status_label = QLabel("")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        layout.addWidget(self.status_label)
        
    def load_models(self):
        """モデルフォルダを読み込む"""
        self.model_tree.clear()
        
        if not MODELS_DIR.exists():
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            return
        
        model_count = 0
        total_size = 0
        
        for model_dir in sorted(MODELS_DIR.iterdir()):
            if model_dir.is_dir():
                item = self.create_model_item(model_dir)
                self.model_tree.addTopLevelItem(item)
                model_count += 1
                total_size += self.get_folder_size(model_dir)
        
        # ステータス更新
        size_str = self.format_size(total_size)
        self.status_label.setText(f"合計: {model_count} モデル ({size_str})")
        
    def create_model_item(self, model_dir):
        """モデルアイテムを作成"""
        item = QTreeWidgetItem()
        
        # モデル名
        item.setText(0, model_dir.name)
        item.setData(0, Qt.ItemDataRole.UserRole, str(model_dir))
        
        # 作成日時
        created_time = datetime.fromtimestamp(model_dir.stat().st_ctime)
        item.setText(1, created_time.strftime("%Y-%m-%d %H:%M"))
        
        # サイズ
        size = self.get_folder_size(model_dir)
        item.setText(2, self.format_size(size))
        
        # 状態（weights/best.ptの存在で判定）
        best_weight = model_dir / "weights" / "best.pt"
        if best_weight.exists():
            item.setText(3, "✓ 完了")
            item.setForeground(3, Qt.GlobalColor.darkGreen)
        else:
            # trainディレクトリもチェック
            train_best = model_dir / "train" / "weights" / "best.pt"
            if train_best.exists():
                item.setText(3, "✓ 完了")
                item.setForeground(3, Qt.GlobalColor.darkGreen)
            else:
                item.setText(3, "⚠ 不完全")
                item.setForeground(3, Qt.GlobalColor.darkYellow)
        
        # ツールチップに詳細情報を設定
        tooltip = f"パス: {model_dir}\n"
        tooltip += f"作成日時: {created_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        tooltip += f"サイズ: {self.format_size(size)}"
        item.setToolTip(0, tooltip)
        
        return item
        
    def get_folder_size(self, folder):
        """フォルダのサイズを取得"""
        total = 0
        try:
            for entry in folder.rglob('*'):
                if entry.is_file():
                    total += entry.stat().st_size
        except:
            pass
        return total
        
    def format_size(self, size):
        """サイズを人間が読める形式にフォーマット"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024.0:
                return f"{size:.1f} {unit}"
            size /= 1024.0
        return f"{size:.1f} TB"
        
    def show_context_menu(self, position):
        """コンテキストメニューを表示"""
        item = self.model_tree.itemAt(position)
        if not item:
            return
            
        menu = QMenu(self)
        
        # 詳細情報
        info_action = menu.addAction("詳細情報")
        info_action.triggered.connect(self.show_model_info)
        
        menu.addSeparator()
        
        # 名前変更
        rename_action = menu.addAction("名前変更")
        rename_action.triggered.connect(self.rename_model)
        
        # 削除
        delete_action = menu.addAction("削除")
        delete_action.triggered.connect(self.delete_model)
        
        menu.addSeparator()
        
        # Finderで開く
        open_action = menu.addAction("Finderで開く")
        open_action.triggered.connect(self.open_in_finder)
        
        menu.exec(self.model_tree.mapToGlobal(position))
        
    def on_item_double_clicked(self, item, column):
        """アイテムがダブルクリックされたとき"""
        self.show_model_info()
        
    def on_selection_changed(self):
        """選択が変更されたとき"""
        item = self.model_tree.currentItem()
        if item:
            model_path = item.data(0, Qt.ItemDataRole.UserRole)
            self.model_selected.emit(model_path)
            
    def refresh_models(self):
        """モデルリストを更新"""
        # 現在の選択を保存
        current_item = self.model_tree.currentItem()
        current_name = current_item.text(0) if current_item else None
        
        # リロード
        self.load_models()
        
        # 選択を復元
        if current_name:
            for i in range(self.model_tree.topLevelItemCount()):
                item = self.model_tree.topLevelItem(i)
                if item.text(0) == current_name:
                    self.model_tree.setCurrentItem(item)
                    break
                    
    def delete_model(self):
        """選択されたモデルを削除"""
        item = self.model_tree.currentItem()
        if not item:
            QMessageBox.warning(self, "警告", "モデルを選択してください")
            return
            
        model_path = Path(item.data(0, Qt.ItemDataRole.UserRole))
        model_name = model_path.name
        
        # 確認ダイアログ
        reply = QMessageBox.question(
            self,
            "削除の確認",
            f"モデル「{model_name}」を削除しますか？\nこの操作は取り消せません。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            try:
                shutil.rmtree(model_path)
                self.refresh_models()
                self.notify_auto_save(f"モデル「{model_name}」を削除しました")
            except Exception as e:
                QMessageBox.critical(self, "エラー", f"削除に失敗しました:\n{str(e)}")
                
    def rename_model(self):
        """選択されたモデルの名前を変更"""
        item = self.model_tree.currentItem()
        if not item:
            QMessageBox.warning(self, "警告", "モデルを選択してください")
            return
            
        model_path = Path(item.data(0, Qt.ItemDataRole.UserRole))
        old_name = model_path.name
        
        # 新しい名前を入力
        new_name, ok = QInputDialog.getText(
            self,
            "名前変更",
            "新しいモデル名:",
            text=old_name
        )
        
        if ok and new_name and new_name != old_name:
            # 無効な文字をチェック
            invalid_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
            if any(char in new_name for char in invalid_chars):
                QMessageBox.warning(
                    self, 
                    "警告", 
                    "モデル名に使用できない文字が含まれています:\n/ \\ : * ? \" < > |"
                )
                return
                
            new_path = model_path.parent / new_name
            
            # 既存のフォルダをチェック
            if new_path.exists():
                QMessageBox.warning(self, "警告", f"「{new_name}」という名前のモデルは既に存在します")
                return
                
            try:
                model_path.rename(new_path)
                self.refresh_models()
                self.notify_auto_save(f"モデル名を「{old_name}」から「{new_name}」に変更しました")
            except Exception as e:
                QMessageBox.critical(self, "エラー", f"名前変更に失敗しました:\n{str(e)}")
                
    def show_model_info(self):
        """モデルの詳細情報を表示"""
        item = self.model_tree.currentItem()
        if not item:
            QMessageBox.warning(self, "警告", "モデルを選択してください")
            return
            
        model_path = item.data(0, Qt.ItemDataRole.UserRole)
        dialog = ModelInfoDialog(model_path, self)
        dialog.exec()
        
    def open_in_finder(self):
        """Finderでモデルフォルダを開く"""
        item = self.model_tree.currentItem()
        if not item:
            QMessageBox.warning(self, "警告", "モデルを選択してください")
            return
            
        model_path = item.data(0, Qt.ItemDataRole.UserRole)
        
        import subprocess
        import platform
        
        if platform.system() == 'Darwin':  # macOS
            subprocess.run(['open', model_path])
        elif platform.system() == 'Windows':
            subprocess.run(['explorer', model_path])
        else:  # Linux
            subprocess.run(['xdg-open', model_path])
            
    def notify_auto_save(self, message):
        """自動保存の通知を送る"""
        parent = self.parent()
        while parent and not hasattr(parent, 'show_save_notification'):
            parent = parent.parent()
            
        if parent and hasattr(parent, 'show_save_notification'):
            parent.show_save_notification(message)