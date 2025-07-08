from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                               QTabWidget, QPushButton, QLabel, QStatusBar,
                               QToolBar, QMessageBox, QDialog, QSizePolicy)
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QAction, QIcon

from ..config import ensure_directories, DEFAULT_CONFIG
from ..utils.device_utils import get_device, get_system_info
from pathlib import Path

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO モザイク処理アプリケーション")
        self.setGeometry(100, 100, 1200, 800)
        
        # ウィンドウサイズを可変にする
        self.setMinimumSize(800, 600)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        ensure_directories()
        self.init_ui()
        self.setup_status_bar()
        self.show_device_info()
        self.setup_auto_save()
        self.load_app_settings()
        
        # タブのスタイルを設定
        self.tab_widget.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #cccccc;
                background-color: white;
            }
            QTabBar::tab {
                padding: 8px 16px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: white;
                border-bottom: 2px solid #3b82f6;
            }
        """)
    
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        central_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout = QVBoxLayout(central_widget)
        
        toolbar = self.create_toolbar()
        
        self.tab_widget = QTabWidget()
        self.tab_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self.tab_widget)
        
        from .annotation_widget import AnnotationWidget
        from .training_widget import TrainingWidget
        from .inference_widget import InferenceWidget
        from .dataset_manager_widget import DatasetManagerWidget
        from .models_manager_widget import ModelsManagerWidget
        
        self.annotation_tab = AnnotationWidget()
        self.training_tab = TrainingWidget()
        self.inference_tab = InferenceWidget()
        self.dataset_tab = DatasetManagerWidget()
        self.models_tab = ModelsManagerWidget()
        
        self.tab_widget.addTab(self.annotation_tab, "アノテーション")
        self.tab_widget.addTab(self.training_tab, "学習")
        self.tab_widget.addTab(self.inference_tab, "推論・モザイク処理")
        self.tab_widget.addTab(self.dataset_tab, "データセット管理")
        self.tab_widget.addTab(self.models_tab, "モデル管理")
    
    def create_toolbar(self):
        toolbar = QToolBar()
        self.addToolBar(toolbar)
        
        # 最近使用したデータセット
        recent_datasets_action = QAction("最近のデータセット", self)
        recent_datasets_action.triggered.connect(self.show_recent_datasets)
        toolbar.addAction(recent_datasets_action)
        
        toolbar.addSeparator()
        
        settings_action = QAction("設定", self)
        settings_action.triggered.connect(self.open_settings)
        toolbar.addAction(settings_action)
        
        toolbar.addSeparator()
        
        help_action = QAction("使い方", self)
        help_action.triggered.connect(self.open_help)
        toolbar.addAction(help_action)
        
        license_action = QAction("ライセンス", self)
        license_action.triggered.connect(self.open_license)
        toolbar.addAction(license_action)
        
        return toolbar
    
    def setup_status_bar(self):
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        self.device_label = QLabel()
        self.status_bar.addPermanentWidget(self.device_label)
    
    def show_device_info(self):
        device, device_name = get_device()
        self.device_label.setText(f"デバイス: {device_name}")
    
    def setup_auto_save(self):
        """自動保存の通知用タイマーを設定"""
        self.save_notification_timer = QTimer()
        self.save_notification_timer.timeout.connect(self.hide_save_notification)
    
    def show_save_notification(self, message="自動保存しました"):
        """自動保存の通知を表示"""
        self.status_bar.showMessage(message, 3000)  # 3秒間表示
    
    def hide_save_notification(self):
        """自動保存の通知を非表示"""
        self.status_bar.clearMessage()
    
    def load_app_settings(self):
        """アプリケーション設定を読み込む"""
        from ..utils.settings_manager import SettingsManager
        self.settings_manager = SettingsManager()
        
        # ウィンドウジオメトリを復元
        geometry = self.settings_manager.get_window_geometry()
        if geometry:
            try:
                self.setGeometry(*geometry)
            except:
                pass
        
        # 最後のタブを復元
        last_tab = self.settings_manager.get_last_tab_index()
        if 0 <= last_tab < self.tab_widget.count():
            self.tab_widget.setCurrentIndex(last_tab)
        
        # 最後に使用したデータセットを読み込む
        last_dataset = self.settings_manager.get_last_dataset()
        if last_dataset and Path(last_dataset).exists():
            self.annotation_tab.load_dataset(Path(last_dataset))
            # 学習タブにも同じデータセットを設定
            self.training_tab.dataset_path = Path(last_dataset)
            self.training_tab.validate_and_load_dataset()
        
        # タブ変更時の処理を追加
        self.tab_widget.currentChanged.connect(self.on_tab_changed)
    
    def on_tab_changed(self, index):
        """タブが変更されたときの処理"""
        self.settings_manager.save_last_tab_index(index)
    
    def closeEvent(self, event):
        """アプリケーション終了時の処理"""
        # ウィンドウジオメトリを保存
        geometry = [self.x(), self.y(), self.width(), self.height()]
        self.settings_manager.save_window_geometry(geometry)
        
        # 現在のデータセットを保存
        if hasattr(self.annotation_tab, 'dataset_path') and self.annotation_tab.dataset_path:
            self.settings_manager.save_last_dataset(str(self.annotation_tab.dataset_path))
        
        # アノテーションの自動保存
        self.annotation_tab.closeEvent(event)
        
        self.show_save_notification("設定を保存しました")
        
        event.accept()
    
    def show_recent_datasets(self):
        from .recent_datasets_dialog import RecentDatasetsDialog
        dialog = RecentDatasetsDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted and dialog.selected_dataset:
            # 選択されたデータセットをアノテーションタブに設定
            self.annotation_tab.load_dataset(dialog.selected_dataset)
    
    def open_settings(self):
        QMessageBox.information(self, "設定", "設定画面を開きます")
    
    def open_help(self):
        from .help_dialog import HelpDialog
        dialog = HelpDialog(self)
        dialog.exec()
    
    def open_license(self):
        from .license_dialog import LicenseDialog
        dialog = LicenseDialog(self)
        dialog.exec()