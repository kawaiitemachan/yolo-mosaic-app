#!/bin/bash

# YOLOMosaicApp ビルドスクリプト

echo "YOLOMosaicApp のビルドを開始します..."

# PyInstallerがインストールされているか確認
if ! command -v pyinstaller &> /dev/null
then
    echo "PyInstallerがインストールされていません。インストールしています..."
    pip install pyinstaller
fi

# distとbuildディレクトリをクリーンアップ
echo "古いビルドファイルを削除しています..."
rm -rf dist build

# --onefileオプションでビルド
echo "アプリケーションをビルドしています..."
pyinstaller --onefile \
    --windowed \
    --name="YOLOMosaicApp" \
    --add-data "src:src" \
    --hidden-import="PySide6" \
    --hidden-import="ultralytics" \
    --hidden-import="torch" \
    --hidden-import="torchvision" \
    --hidden-import="cv2" \
    --hidden-import="numpy" \
    --hidden-import="PIL" \
    --hidden-import="yaml" \
    main.py

# macOSの場合、.appバンドルを作成
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "macOS用の.appバンドルを作成しています..."
    # 既に--windowedオプションで.appが作成されているはず
fi

echo "ビルドが完了しました！"
echo "実行ファイルは dist/ ディレクトリにあります。"

# 実行権限を付与（macOS/Linux）
if [[ "$OSTYPE" == "darwin"* ]] || [[ "$OSTYPE" == "linux-gnu"* ]]; then
    chmod +x dist/YOLOMosaicApp
fi