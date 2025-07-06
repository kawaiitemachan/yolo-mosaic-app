#!/usr/bin/env python3
"""
YOLO モザイク処理アプリケーションのテスト起動スクリプト

使用方法:
    python test_app.py
"""

import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from PySide6.QtWidgets import QApplication
from src.ui.main_window import MainWindow

def main():
    """アプリケーションのメインエントリーポイント"""
    app = QApplication(sys.argv)
    app.setApplicationName("YOLO Mosaic App")
    app.setOrganizationName("YourOrganization")
    
    # メインウィンドウを作成して表示
    window = MainWindow()
    window.show()
    
    # アプリケーションを実行
    sys.exit(app.exec())

if __name__ == "__main__":
    main()