import sys
from PySide6.QtWidgets import QApplication
from src.ui.main_window import MainWindow
from src.utils.app_paths import ensure_directories

def main():
    # アプリケーションに必要なディレクトリを作成
    ensure_directories()
    
    app = QApplication(sys.argv)
    app.setApplicationName("YOLO Mosaic App")
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()