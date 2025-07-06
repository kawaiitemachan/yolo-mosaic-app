from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
                               QLabel, QProgressBar, QTextEdit, QDialogButtonBox)
from PySide6.QtCore import Qt, QThread, Signal
from pathlib import Path

from ..inference.mosaic import batch_process_images

class BatchProcessThread(QThread):
    progress = Signal(int, int, str)
    finished = Signal(list)
    
    def __init__(self, image_paths, model_path, output_dir, config):
        super().__init__()
        self.image_paths = image_paths
        self.model_path = model_path
        self.output_dir = output_dir
        self.config = config
        
    def run(self):
        from ultralytics import YOLO
        from ..inference.mosaic import apply_mosaic_to_regions
        import cv2
        from PIL import Image as PILImage
        
        model = YOLO(self.model_path)
        output_dir = Path(self.output_dir)
        output_dir.mkdir(exist_ok=True)
        
        results = []
        
        for idx, image_path in enumerate(self.image_paths):
            image_path = Path(image_path)
            self.progress.emit(idx + 1, len(self.image_paths), f"処理中: {image_path.name}")
            
            try:
                png_info = None
                if self.config['preserve_png_info'] and image_path.suffix.lower() == '.png':
                    try:
                        pil_img = PILImage.open(str(image_path))
                        png_info = pil_img.info
                    except:
                        png_info = None
                
                image = cv2.imread(str(image_path))
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                predictions = model(str(image_path), 
                                  conf=self.config['confidence'], 
                                  iou=self.config['iou'])
                
                detections = []
                for r in predictions:
                    if r.masks is not None:
                        for i, mask in enumerate(r.masks.data):
                            mask_np = mask.cpu().numpy()
                            mask_resized = cv2.resize(mask_np, (image.shape[1], image.shape[0]))
                            
                            cls = int(r.boxes.cls[i])
                            conf = float(r.boxes.conf[i])
                            label = model.names[cls]
                            
                            detections.append({
                                'mask': mask_resized > 0.5,
                                'label': label,
                                'confidence': conf
                            })
                
                if detections:
                    processed = apply_mosaic_to_regions(
                        image_rgb, detections, 
                        self.config['blur_type'], 
                        self.config['strength']
                    )
                    
                    output_path = output_dir / f"mosaic_{image_path.name}"
                    
                    if self.config['preserve_png_info'] and png_info and image_path.suffix.lower() == '.png':
                        pil_processed = PILImage.fromarray(processed)
                        pil_processed.save(str(output_path), pnginfo=png_info)
                    else:
                        processed_bgr = cv2.cvtColor(processed, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(str(output_path), processed_bgr)
                    
                    results.append({
                        'input': str(image_path),
                        'output': str(output_path),
                        'detections': len(detections),
                        'status': 'success'
                    })
                else:
                    results.append({
                        'input': str(image_path),
                        'output': None,
                        'detections': 0,
                        'status': 'no_detections'
                    })
                    
            except Exception as e:
                results.append({
                    'input': str(image_path),
                    'output': None,
                    'detections': 0,
                    'status': f'error: {str(e)}'
                })
        
        self.finished.emit(results)

class BatchProcessDialog(QDialog):
    def __init__(self, image_paths, model_path, output_dir, config, parent=None):
        super().__init__(parent)
        self.setWindowTitle("バッチ処理実行中")
        self.setModal(True)
        self.resize(600, 400)
        
        self.image_paths = image_paths
        self.model_path = model_path
        self.output_dir = output_dir
        self.config = config
        
        self.init_ui()
        self.start_processing()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        self.status_label = QLabel("処理を開始しています...")
        layout.addWidget(self.status_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximum(len(self.image_paths))
        layout.addWidget(self.progress_bar)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text)
        
        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)
        
    def start_processing(self):
        self.thread = BatchProcessThread(
            self.image_paths, 
            self.model_path, 
            self.output_dir, 
            self.config
        )
        self.thread.progress.connect(self.update_progress)
        self.thread.finished.connect(self.on_finished)
        self.thread.start()
        
    def update_progress(self, current, total, message):
        self.progress_bar.setValue(current)
        self.status_label.setText(f"{current}/{total} - {message}")
        self.log_text.append(message)
        
    def on_finished(self, results):
        success_count = sum(1 for r in results if r['status'] == 'success')
        no_detection_count = sum(1 for r in results if r['status'] == 'no_detections')
        error_count = sum(1 for r in results if r['status'].startswith('error'))
        
        summary = f"\n処理完了:\n"
        summary += f"  成功: {success_count}\n"
        summary += f"  検出なし: {no_detection_count}\n"
        summary += f"  エラー: {error_count}\n"
        
        self.log_text.append(summary)
        
        self.button_box.clear()
        self.button_box.addButton(QDialogButtonBox.StandardButton.Ok)
        self.button_box.accepted.connect(self.accept)
        
        self.results = results