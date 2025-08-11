import sys
import io
from PySide6.QtWidgets import QApplication
from src.ui.main_window import MainWindow
from src.utils.app_paths import ensure_directories

def main():
    # パッケージング環境で標準出力が存在しない場合の対策
    # PyInstallerやpy2exeでGUIモードでビルドされた場合、
    # sys.stdout/stderrがNoneになることがあるため
    if sys.stdout is None:
        sys.stdout = io.StringIO()
    if sys.stderr is None:
        sys.stderr = io.StringIO()
    
    # アプリケーションに必要なディレクトリを作成
    ensure_directories()
    
    app = QApplication(sys.argv)
    app.setApplicationName("YOLO Mosaic App")
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()