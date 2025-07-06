from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                               QLabel, QSpinBox, QComboBox, QPlainTextEdit,
                               QProgressBar, QGroupBox, QFileDialog,
                               QCheckBox, QDialog, QMessageBox, QApplication)
from PySide6.QtCore import Qt, QThread, Signal, QUrl
from PySide6.QtGui import QDragEnterEvent, QDropEvent, QTextCursor
import yaml
from pathlib import Path
import sys
import io
import contextlib
import logging

from ..config import MODELS_DIR, DEFAULT_CONFIG
from ..utils.device_utils import get_device

class OutputCapture:
    """標準出力とエラー出力をキャプチャ"""
    def __init__(self, signal, is_stderr=False):
        self.signal = signal
        self.terminal = sys.stderr if is_stderr else sys.stdout
        self.is_stderr = is_stderr
        
    def write(self, message):
        self.terminal.write(message)
        if message.strip():
            self.signal.emit(message.strip())
            
    def flush(self):
        self.terminal.flush()
        
    def isatty(self):
        return False

class LogHandler(logging.Handler):
    """ログ出力をキャプチャするハンドラー"""
    def __init__(self, signal):
        super().__init__()
        self.signal = signal
        
    def emit(self, record):
        msg = self.format(record)
        if msg.strip():
            self.signal.emit(msg)

class TrainingThread(QThread):
    progress = Signal(str)
    finished = Signal(bool, str)
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Mac環境でのマルチプロセッシング問題を回避
        import platform
        if platform.system() == 'Darwin':
            import multiprocessing
            # macOSでのデフォルトは'spawn'だが、'fork'の方が安定する場合がある
            try:
                multiprocessing.set_start_method('fork', force=True)
            except RuntimeError:
                pass  # 既に設定されている場合は無視
        
    def run(self):
        try:
            from ultralytics import YOLO
            
            # 標準出力とエラー出力を保存
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            
            # 出力をキャプチャ
            sys.stdout = OutputCapture(self.progress, is_stderr=False)
            sys.stderr = OutputCapture(self.progress, is_stderr=True)
            
            # Ultralyticsのロガーをセットアップ
            ultralytics_logger = logging.getLogger("ultralytics")
            ultralytics_logger.setLevel(logging.INFO)
            
            # 既存のハンドラーを削除
            for handler in ultralytics_logger.handlers[:]:
                ultralytics_logger.removeHandler(handler)
            
            # カスタムハンドラーを追加
            log_handler = LogHandler(self.progress)
            log_handler.setFormatter(logging.Formatter('%(message)s'))
            ultralytics_logger.addHandler(log_handler)
            
            # YOLOのロガーも同様に設定
            yolo_logger = logging.getLogger("yolo")
            yolo_logger.setLevel(logging.INFO)
            for handler in yolo_logger.handlers[:]:
                yolo_logger.removeHandler(handler)
            yolo_logger.addHandler(log_handler)
            
            try:
                self.progress.emit("モデルを読み込み中...")
                model = YOLO(self.config['model'])
                
                device, device_name = get_device()
                if self.config.get('device', 'auto') != 'auto':
                    device = self.config['device']
                    
                # Mac環境でMPSデバイスの場合の注意喚起
                import platform
                if platform.system() == 'Darwin' and 'mps' in str(device):
                    self.progress.emit("警告: Mac環境でのMPS使用は不安定な場合があります。")
                    self.progress.emit("エラーが発生した場合は、CPUモードをお試しください。")
                
                self.progress.emit(f"学習を開始します (デバイス: {device})")
                
                # 学習パラメータの設定
                import platform
                train_kwargs = {
                    'data': self.config['data_yaml'],
                    'epochs': self.config['epochs'],
                    'imgsz': self.config['imgsz'],
                    'batch': self.config['batch_size'],
                    'patience': self.config['patience'],
                    'device': device,
                    'project': str(MODELS_DIR),
                    'name': self.config['project_name'],
                    'exist_ok': True,
                    'verbose': True,  # 詳細なログ出力を有効化
                    'workers': 0,  # マルチプロセッシングを無効化（バスエラー対策）
                }
                
                # 追加のロガー設定
                import torch
                torch_logger = logging.getLogger("torch")
                torch_logger.setLevel(logging.INFO)
                torch_logger.addHandler(log_handler)
                
                # Mac環境では追加の安全対策
                if platform.system() == 'Darwin':
                    train_kwargs['plots'] = False  # プロットを無効化
                    train_kwargs['cache'] = False  # キャッシュを無効化
                    # MPSデバイスで問題が発生した場合はCPUにフォールバック
                    if 'mps' in str(device):
                        try:
                            # まずMPSで試行
                            self.progress.emit("MPSデバイスで学習を開始します...")
                            results = model.train(**train_kwargs)
                        except Exception as mps_error:
                            self.progress.emit(f"MPSエラー: {str(mps_error)}")
                            self.progress.emit("CPUモードに切り替えて再試行します...")
                            train_kwargs['device'] = 'cpu'
                            results = model.train(**train_kwargs)
                    else:
                        results = model.train(**train_kwargs)
                else:
                    # Mac以外の環境
                    results = model.train(**train_kwargs)
                
                self.finished.emit(True, "学習が完了しました")
                
            finally:
                # 標準出力とエラー出力を復元
                sys.stdout = old_stdout
                sys.stderr = old_stderr
                
                # ロガーのハンドラーをクリーンアップ
                for logger_name in ["ultralytics", "yolo"]:
                    logger = logging.getLogger(logger_name)
                    for handler in logger.handlers[:]:
                        if isinstance(handler, LogHandler):
                            logger.removeHandler(handler)
                
        except Exception as e:
            # エラーが発生した場合も出力を復元
            if 'old_stdout' in locals():
                sys.stdout = old_stdout
            if 'old_stderr' in locals():
                sys.stderr = old_stderr
            self.finished.emit(False, f"エラー: {str(e)}")

class DatasetDropArea(QWidget):
    """データセットのドラッグ＆ドロップエリア"""
    dataset_dropped = Signal(str)
    
    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.setMinimumHeight(80)
        self.setStyleSheet("""
            DatasetDropArea {
                border: 2px dashed #ccc;
                border-radius: 10px;
                background-color: #f8f9fa;
            }
            DatasetDropArea:hover {
                border-color: #3b82f6;
                background-color: #eff6ff;
            }
        """)
        
        layout = QVBoxLayout(self)
        self.label = QLabel("データセットフォルダをここにドラッグ＆ドロップ\nまたは")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.label)
        
    def dragEnterEvent(self, event: QDragEnterEvent):
        """ドラッグエンターイベント"""
        if event.mimeData().hasUrls():
            # URLが1つでフォルダの場合のみ受け付ける
            urls = event.mimeData().urls()
            if len(urls) == 1:
                path = Path(urls[0].toLocalFile())
                if path.is_dir():
                    event.acceptProposedAction()
                    self.setStyleSheet("""
                        DatasetDropArea {
                            border: 2px solid #3b82f6;
                            border-radius: 10px;
                            background-color: #dbeafe;
                        }
                    """)
                    return
        event.ignore()
    
    def dragLeaveEvent(self, event):
        """ドラッグリーブイベント"""
        self.setStyleSheet("""
            DatasetDropArea {
                border: 2px dashed #ccc;
                border-radius: 10px;
                background-color: #f8f9fa;
            }
        """)
    
    def dropEvent(self, event: QDropEvent):
        """ドロップイベント"""
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if len(urls) == 1:
                path = Path(urls[0].toLocalFile())
                if path.is_dir():
                    event.acceptProposedAction()
                    self.dataset_dropped.emit(str(path))
                    # スタイルを元に戻す
                    self.setStyleSheet("""
                        DatasetDropArea {
                            border: 2px dashed #ccc;
                            border-radius: 10px;
                            background-color: #f8f9fa;
                        }
                    """)
                    return
        event.ignore()

class TrainingWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.dataset_path = None
        self.data_yaml_path = None
        self.init_ui()
        self.training_thread = None
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        data_group = self.create_data_group()
        layout.addWidget(data_group)
        
        model_group = self.create_model_group()
        layout.addWidget(model_group)
        
        training_group = self.create_training_group()
        layout.addWidget(training_group)
        
        self.start_button = QPushButton("学習開始")
        self.start_button.clicked.connect(self.start_training)
        layout.addWidget(self.start_button)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)
        
        self.log_text = QPlainTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumBlockCount(1000)  # 最大1000行まで保持
        self.log_text.setStyleSheet("""
            QPlainTextEdit {
                font-family: 'Monaco', 'Menlo', 'Consolas', 'Courier New', monospace;
                font-size: 12px;
                background-color: #1e1e1e;
                color: #d4d4d4;
                border: 1px solid #3e3e3e;
                padding: 10px;
            }
        """)
        layout.addWidget(self.log_text)
        
    def create_data_group(self):
        group = QGroupBox("データセット")
        layout = QVBoxLayout(group)
        
        # ドラッグ＆ドロップエリア
        self.drop_area = DatasetDropArea()
        self.drop_area.dataset_dropped.connect(self.on_dataset_dropped)
        
        # ドロップエリアのレイアウトを取得（既に作成済み）
        drop_layout = self.drop_area.layout()
        
        # ボタンを作成して追加
        select_dataset_btn = QPushButton("データセット選択")
        select_dataset_btn.clicked.connect(self.select_dataset)
        select_dataset_btn.setStyleSheet("""
            QPushButton {
                background-color: #3b82f6;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2563eb;
            }
        """)
        drop_layout.addWidget(select_dataset_btn, alignment=Qt.AlignmentFlag.AlignCenter)
        drop_layout.addStretch()
        
        layout.addWidget(self.drop_area)
        
        # データセット情報表示
        self.dataset_info_label = QLabel("データセット未選択")
        self.dataset_info_label.setWordWrap(True)
        self.dataset_info_label.setStyleSheet("QLabel { padding: 10px; background-color: #f3f4f6; border-radius: 5px; }")
        layout.addWidget(self.dataset_info_label)
        
        # データ分布表示
        self.data_distribution_label = QLabel("")
        self.data_distribution_label.setWordWrap(True)
        layout.addWidget(self.data_distribution_label)
        
        return group
    
    def create_model_group(self):
        group = QGroupBox("モデル設定")
        layout = QVBoxLayout(group)
        
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("ベースモデル:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "yolo11n-seg.pt",
            "yolo11s-seg.pt",
            "yolo11m-seg.pt",
            "yolo11l-seg.pt",
            "yolo11x-seg.pt"
        ])
        model_layout.addWidget(self.model_combo)
        layout.addLayout(model_layout)
        
        device_layout = QHBoxLayout()
        device_layout.addWidget(QLabel("デバイス:"))
        self.device_combo = QComboBox()
        self.device_combo.addItems(["auto", "cuda", "mps", "cpu"])
        device_layout.addWidget(self.device_combo)
        layout.addLayout(device_layout)
        
        return group
    
    def create_training_group(self):
        group = QGroupBox("学習パラメータ")
        layout = QVBoxLayout(group)
        
        epochs_layout = QHBoxLayout()
        epochs_layout.addWidget(QLabel("エポック数:"))
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(DEFAULT_CONFIG["training"]["epochs"])
        epochs_layout.addWidget(self.epochs_spin)
        layout.addLayout(epochs_layout)
        
        batch_layout = QHBoxLayout()
        batch_layout.addWidget(QLabel("バッチサイズ:"))
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 128)
        self.batch_spin.setValue(DEFAULT_CONFIG["training"]["batch_size"])
        batch_layout.addWidget(self.batch_spin)
        layout.addLayout(batch_layout)
        
        imgsz_layout = QHBoxLayout()
        imgsz_layout.addWidget(QLabel("画像サイズ:"))
        self.imgsz_spin = QSpinBox()
        self.imgsz_spin.setRange(32, 1280)
        self.imgsz_spin.setSingleStep(32)
        self.imgsz_spin.setValue(DEFAULT_CONFIG["training"]["imgsz"])
        imgsz_layout.addWidget(self.imgsz_spin)
        layout.addLayout(imgsz_layout)
        
        patience_layout = QHBoxLayout()
        patience_layout.addWidget(QLabel("Early Stopping:"))
        self.patience_spin = QSpinBox()
        self.patience_spin.setRange(0, 300)
        self.patience_spin.setValue(DEFAULT_CONFIG["training"]["patience"])
        patience_layout.addWidget(self.patience_spin)
        layout.addLayout(patience_layout)
        
        return group
    
    def select_dataset(self):
        """データセット選択ダイアログを表示"""
        from ..config import DATASETS_DIR
        
        folder = QFileDialog.getExistingDirectory(
            self, 
            "学習用データセットを選択",
            str(DATASETS_DIR) if DATASETS_DIR.exists() else str(Path.home())
        )
        
        if folder:
            self.dataset_path = Path(folder)
            self.validate_and_load_dataset()
    
    def on_dataset_dropped(self, folder_path):
        """データセットフォルダがドロップされたときの処理"""
        self.dataset_path = Path(folder_path)
        self.validate_and_load_dataset()
    
    def validate_and_load_dataset(self):
        """データセットの妥当性を確認して読み込む"""
        if not self.dataset_path:
            return
        
        # data.yamlの存在確認
        yaml_path = self.dataset_path / "data.yaml"
        if not yaml_path.exists():
            QMessageBox.warning(
                self, 
                "警告", 
                "選択されたフォルダにdata.yamlが見つかりません。\n正しいデータセットフォルダを選択してください。"
            )
            self.dataset_path = None
            
            # ドロップエリアの表示を元に戻す
            self.drop_area.label.setText("データセットフォルダをここにドラッグ＆ドロップ\nまたは")
            self.drop_area.setStyleSheet("""
                DatasetDropArea {
                    border: 2px dashed #ccc;
                    border-radius: 10px;
                    background-color: #f8f9fa;
                }
            """)
            return
        
        self.data_yaml_path = str(yaml_path)
        
        try:
            # data.yamlを読み込む
            with open(yaml_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            # データセット情報を表示
            info_text = f"データセット: {self.dataset_path.name}\n"
            info_text += f"クラス数: {data.get('nc', 0)}\n"
            
            classes = data.get('names', [])
            if classes:
                info_text += f"クラス: {', '.join(classes)}"
            
            self.dataset_info_label.setText(info_text)
            
            # データ分布を確認
            self.check_data_distribution(data)
            
            # ドロップエリアの表示を更新
            self.drop_area.label.setText(f"✓ {self.dataset_path.name}\n選択済み")
            self.drop_area.setStyleSheet("""
                DatasetDropArea {
                    border: 2px solid #16a34a;
                    border-radius: 10px;
                    background-color: #f0fdf4;
                }
            """)
            
            # 自動保存通知
            self.notify_auto_save()
            
        except Exception as e:
            QMessageBox.critical(
                self, 
                "エラー", 
                f"data.yaml の読み込みに失敗しました:\n{str(e)}"
            )
            self.dataset_path = None
            self.data_yaml_path = None
            
            # ドロップエリアの表示を元に戻す
            self.drop_area.label.setText("データセットフォルダをここにドラッグ＆ドロップ\nまたは")
            self.drop_area.setStyleSheet("""
                DatasetDropArea {
                    border: 2px dashed #ccc;
                    border-radius: 10px;
                    background-color: #f8f9fa;
                }
            """)
    
    def check_data_distribution(self, data):
        """データセット内のデータ分布を確認"""
        distribution_text = ""
        
        # 訓練データの確認
        train_path = self.dataset_path / "train"
        if train_path.exists():
            train_images = len(list((train_path / "images").glob("*.*"))) if (train_path / "images").exists() else 0
            train_labels = len(list((train_path / "labels").glob("*.txt"))) if (train_path / "labels").exists() else 0
            distribution_text += f"訓練データ: 画像 {train_images} 枚, ラベル {train_labels} 件\n"
        
        # 検証データの確認
        valid_path = self.dataset_path / "valid"
        val_path = self.dataset_path / "val"
        
        if valid_path.exists():
            valid_images = len(list((valid_path / "images").glob("*.*"))) if (valid_path / "images").exists() else 0
            valid_labels = len(list((valid_path / "labels").glob("*.txt"))) if (valid_path / "labels").exists() else 0
            distribution_text += f"検証データ: 画像 {valid_images} 枚, ラベル {valid_labels} 件"
        elif val_path.exists():
            val_images = len(list((val_path / "images").glob("*.*"))) if (val_path / "images").exists() else 0
            val_labels = len(list((val_path / "labels").glob("*.txt"))) if (val_path / "labels").exists() else 0
            distribution_text += f"検証データ: 画像 {val_images} 枚, ラベル {val_labels} 件"
        
        self.data_distribution_label.setText(distribution_text)
        
        # データ不足の警告
        if train_images == 0:
            QMessageBox.warning(
                self,
                "警告",
                "訓練データが見つかりません。\nアノテーションタブでデータを作成してください。"
            )
    
    def notify_auto_save(self):
        """自動保存の通知を送る"""
        parent = self.parent()
        while parent and not hasattr(parent, 'show_save_notification'):
            parent = parent.parent()
        
        if parent and hasattr(parent, 'show_save_notification'):
            parent.show_save_notification("学習設定を自動保存しました")
    
    def start_training(self):
        if not self.data_yaml_path:
            self.log("エラー: データセットを選択してください")
            QMessageBox.warning(self, "警告", "学習を開始する前にデータセットを選択してください")
            return
        
        # データセットの検証
        validation_result = self.validate_dataset()
        if not validation_result["valid"]:
            QMessageBox.critical(
                self,
                "データセットエラー",
                validation_result["message"]
            )
            return
        
        # プロジェクト名をデータセット名から生成
        project_name = f"{self.dataset_path.name}_training"
        
        config = {
            'data_yaml': self.data_yaml_path,
            'model': self.model_combo.currentText(),
            'epochs': self.epochs_spin.value(),
            'batch_size': self.batch_spin.value(),
            'imgsz': self.imgsz_spin.value(),
            'patience': self.patience_spin.value(),
            'device': self.device_combo.currentText(),
            'project_name': project_name
        }
        
        self.start_button.setEnabled(False)
        self.progress_bar.show()
        self.progress_bar.setRange(0, 0)
        
        # ログテキストをクリア
        self.log_text.clear()
        
        self.training_thread = TrainingThread(config)
        self.training_thread.progress.connect(self.log)
        self.training_thread.finished.connect(self.on_training_finished)
        self.training_thread.start()
    
    def log(self, message):
        """ログメッセージを表示"""
        # メッセージを追加
        self.log_text.appendPlainText(message)
        
        # カーソルを最後に移動して、スクロールバーを最下部に
        self.log_text.moveCursor(QTextCursor.MoveOperation.End)
        self.log_text.ensureCursorVisible()
        
        # UIを強制的に更新
        QApplication.processEvents()
    
    def validate_dataset(self):
        """データセットの検証"""
        result = {"valid": True, "message": ""}
        
        if not self.dataset_path or not self.dataset_path.exists():
            result["valid"] = False
            result["message"] = "データセットフォルダが存在しません"
            return result
        
        # data.yamlの確認
        yaml_path = self.dataset_path / "data.yaml"
        if not yaml_path.exists():
            result["valid"] = False
            result["message"] = "data.yamlファイルが見つかりません"
            return result
        
        try:
            with open(yaml_path, 'r') as f:
                data = yaml.safe_load(f)
            
            # 必須フィールドの確認
            if 'names' not in data:
                result["valid"] = False
                result["message"] = "data.yamlにクラス名(names)が定義されていません"
                return result
                
            if 'nc' not in data:
                result["valid"] = False
                result["message"] = "data.yamlにクラス数(nc)が定義されていません"
                return result
        except Exception as e:
            result["valid"] = False
            result["message"] = f"data.yaml読み込みエラー: {str(e)}"
            return result
        
        # train/imagesフォルダの確認
        train_images = self.dataset_path / "train" / "images"
        if not train_images.exists():
            result["valid"] = False
            result["message"] = "train/imagesフォルダが見つかりません"
            return result
        
        # 画像ファイルの確認
        image_files = list(train_images.glob("*.*"))
        if len(image_files) == 0:
            result["valid"] = False
            result["message"] = "train/imagesフォルダに画像がありません"
            return result
        
        # 空のラベルファイルの確認（警告のみ）
        train_labels = self.dataset_path / "train" / "labels"
        if train_labels.exists():
            empty_labels = 0
            for label_file in train_labels.glob("*.txt"):
                if label_file.stat().st_size == 0:
                    empty_labels += 1
            
            if empty_labels > 0:
                self.log(f"注意: {empty_labels}個の空のラベルファイルがあります（背景画像として扱われます）")
        
        return result
    
    def on_training_finished(self, success, message):
        self.start_button.setEnabled(True)
        self.progress_bar.hide()
        self.log(message)
        
        if success:
            self.log(f"学習済みモデルは {MODELS_DIR} に保存されました")