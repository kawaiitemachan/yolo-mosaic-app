# YOLO Mosaic App

YOLOv11を使用したインスタンスセグメンテーションによるモザイク処理アプリケーション

## 概要

このアプリケーションは、Ultralytics YOLOv11のインスタンスセグメンテーション機能を使用して、画像内の特定のオブジェクトを検出し、自動的にモザイク処理を適用するツールです。

### 主な機能

- 📸 **画像アノテーション**: YOLOフォーマットでのセグメンテーションアノテーション作成
- 🎯 **モデル学習**: カスタムデータセットでのYOLOv11モデルの学習
- 🔍 **推論とモザイク処理**: 学習済みモデルを使用した自動モザイク処理
- 📊 **データセット管理**: アノテーションデータの管理と分割
- 🤖 **モデル管理**: 学習済みモデルの管理と選択

## ⚠️ ライセンス情報 - 重要

**このアプリケーションはAGPL-3.0ライセンスです。**

### なぜAGPL-3.0なのか？

このアプリケーションは[Ultralytics YOLO](https://github.com/ultralytics/ultralytics)を使用しており、UltralyticsはAGPL-3.0ライセンスで配布されています。AGPL-3.0は「伝染性」のあるライセンスであり、これを使用するソフトウェアも同じライセンスで配布する必要があります。

### AGPL-3.0の要件

- ✅ **ソースコードの公開**: このリポジトリで完全に公開しています
- ✅ **改変の公開**: フォークや改変版も同じライセンスで公開する必要があります
- ✅ **ネットワーク使用時の公開**: Webサービスとして提供する場合もソースコード公開が必要です

### 商用利用について

商用利用を検討されている場合、以下の選択肢があります：

1. **AGPL-3.0の要件を受け入れる**: ソースコードを公開し続ける
2. **Ultralyticsの商用ライセンスを取得**: [https://www.ultralytics.com/license](https://www.ultralytics.com/license)

詳細は[LICENSE_INFO.md](LICENSE_INFO.md)をご確認ください。

## インストール

### 必要な環境

- Python 3.8以上
- CUDA対応GPU（推奨）

### 手順

1. リポジトリをクローン
```bash
git clone https://github.com/kawaiitemachan/yolo-mosaic-app.git
cd yolo-mosaic-app
```

2. 依存関係をインストール
```bash
pip install -r requirements.txt
```

3. アプリケーションを起動
```bash
python main.py
```

## 使い方

### 1. データセットの準備

- 「アノテーション」タブで画像をアノテーション
- ポリゴンツールを使用してオブジェクトをセグメンテーション
- YOLOフォーマットで自動保存

### 2. モデルの学習

- 「学習」タブでデータセットを選択
- 学習パラメータを設定
- 「学習開始」をクリック

### 3. モザイク処理

- 「推論・モザイク処理」タブで学習済みモデルを選択
- 処理したい画像またはフォルダを選択
- モザイクの種類と強度を設定
- 処理を実行

## ビルド

スタンドアロンアプリケーションとしてビルドする場合：

### Windows
```bash
./build_app_windows.ps1
```

### macOS/Linux
```bash
./build_app.sh
```

## 貢献

プルリクエストを歓迎します！バグ報告や機能要望は[Issues](https://github.com/kawaiitemachan/yolo-mosaic-app/issues)でお願いします。

## 謝辞

- [Ultralytics](https://www.ultralytics.com/) - YOLOv11の実装
- [Qt/PySide6](https://www.qt.io/qt-for-python) - GUI フレームワーク

## ライセンス

このプロジェクトはAGPL-3.0ライセンスで公開されています。詳細は[LICENSE](LICENSE)および[LICENSE_INFO.md](LICENSE_INFO.md)をご確認ください。

---

© 2025 kawaiitemachan - [GitHub](https://github.com/kawaiitemachan)