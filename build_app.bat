@echo off
chcp 65001 >nul
REM YOLOMosaicApp ビルドスクリプト (Windows版)

echo YOLOMosaicApp のビルドを開始します...

REM PyInstallerがインストールされているか確認
python -m pip show pyinstaller >nul 2>&1
if %errorlevel% neq 0 (
    echo PyInstallerがインストールされていません。インストールしています...
    python -m pip install pyinstaller
)

REM distとbuildディレクトリをクリーンアップ
echo 古いビルドファイルを削除しています...
if exist dist rmdir /s /q dist
if exist build rmdir /s /q build

REM --onefileオプションでビルド
echo アプリケーションをビルドしています...
pyinstaller --onefile ^
    --windowed ^
    --name="YOLOMosaicApp" ^
    --add-data "src;src" ^
    --hidden-import="PySide6" ^
    --hidden-import="ultralytics" ^
    --hidden-import="torch" ^
    --hidden-import="torchvision" ^
    --hidden-import="cv2" ^
    --hidden-import="numpy" ^
    --hidden-import="PIL" ^
    --hidden-import="yaml" ^
    main.py

echo ビルドが完了しました！
echo 実行ファイルは dist\ ディレクトリにあります。

pause