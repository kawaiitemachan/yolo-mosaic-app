# YOLO Mosaic App

YOLOモデルのトレーニングデータ作成（アノテーション）、学習、推論（モザイク処理）を行うPySide6アプリケーション

## 主な機能

### 1. アノテーション機能
- ポリゴン形式でのインスタンスセグメンテーション
- 複数クラスのサポート
- 既存ポリゴンの編集（ドラッグで移動）
- アンドゥ・リドゥ機能
- 自動保存機能

### 2. 学習機能
- YOLOv11モデル（n, s, m, l, x）のサポート
- GPU自動選択（Windows RTX → Mac Metal → CPU）
- リアルタイム学習ログ表示
- Mac環境でのバスエラー対策済み

### 3. 推論・モザイク処理
- ガウシアンブラー
- ブラー
- ピクセレート
- 黒塗りつぶし
- 白塗りつぶし
- タイルモザイク（ピクセルサイズ指定可能）
- フォルダ単位でのバッチ処理
- PNG metadata保存オプション

## 動作環境

- Python 3.8以上
- PySide6
- Ultralytics (YOLOv11)
- OpenCV
- PyTorch

## インストール

```bash
git clone https://github.com/kawaiitemachan/yolo-mosaic-app.git
cd yolo-mosaic-app
pip install -r requirements.txt
```

## 使用方法

### 開発環境での実行

```bash
python main.py
```

### スタンドアロンアプリケーションの作成

PyInstallerを使用して単一の実行ファイルを作成できます：

```bash
# ビルドスクリプトを使用
./build_app.sh

# または手動でビルド
pyinstaller --onefile --windowed --name="YOLOMosaicApp" main.py
```

ビルドされたアプリケーションは `dist/` ディレクトリに作成されます。

## ディレクトリ構造

アプリケーションは初回起動時に以下のディレクトリを自動的に作成します：

```
YOLOMosaicApp（実行ファイル）
├── data/
│   ├── images/         # アノテーション用画像
│   ├── annotations/    # アノテーションデータ
│   └── models/         # 学習済みモデル
└── datasets/           # データセット
```

## 注意事項

- PyInstallerでビルドした場合も、データフォルダは実行ファイルと同じ階層に作成されます
- 学習済みモデル（.ptファイル）や大きなデータセットはGitには含まれません

## ライセンス

MIT License

