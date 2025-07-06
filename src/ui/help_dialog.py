from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QTextBrowser,
                               QListWidget, QSplitter, QDialogButtonBox,
                               QListWidgetItem)
from PySide6.QtCore import Qt, QSize

class HelpDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("YOLO モザイク処理アプリケーション - 使い方")
        self.setModal(True)
        self.resize(900, 600)
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # 左側：目次
        self.contents_list = QListWidget()
        self.contents_list.setMaximumWidth(250)
        self.add_contents()
        self.contents_list.currentItemChanged.connect(self.on_content_changed)
        
        # 右側：内容表示
        self.help_browser = QTextBrowser()
        self.help_browser.setOpenExternalLinks(False)
        
        splitter.addWidget(self.contents_list)
        splitter.addWidget(self.help_browser)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        
        layout.addWidget(splitter)
        
        # ボタン
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)
        
        # 初期表示
        self.contents_list.setCurrentRow(0)
        
    def add_contents(self):
        """目次を追加"""
        contents = [
            ("はじめに", "introduction"),
            ("データセット作成と自動保存", "dataset"),
            ("アノテーション機能", "annotation"),
            ("学習機能", "training"),
            ("推論・モザイク処理", "inference"),
            ("データセット管理タブ", "dataset_manager"),
            ("モデル管理", "models"),
            ("バッチ処理", "batch"),
            ("設定とカスタマイズ", "settings"),
            ("トラブルシューティング", "troubleshooting")
        ]
        
        for title, tag in contents:
            item = QListWidgetItem(title)
            item.setData(Qt.ItemDataRole.UserRole, tag)
            self.contents_list.addItem(item)
    
    def on_content_changed(self, current, previous):
        """選択された項目に応じて内容を表示"""
        if not current:
            return
            
        tag = current.data(Qt.ItemDataRole.UserRole)
        content = self.get_help_content(tag)
        self.help_browser.setHtml(content)
    
    def get_help_content(self, tag):
        """ヘルプコンテンツを取得"""
        # QTextBrowser互換のシンプルなスタイル
        style = """
        <style>
        body { font-family: Arial, sans-serif; margin: 10px; }
        h2 { color: #2563eb; margin-top: 20px; }
        h3 { color: #1e40af; margin-top: 15px; }
        p { margin: 10px 0; }
        ul { margin: 10px 0; padding-left: 20px; }
        li { margin: 5px 0; }
        table { border-collapse: collapse; margin: 10px 0; }
        td { padding: 8px; border: 1px solid #e5e7eb; }
        </style>
        """
        
        contents = {
            "introduction": style + """
<h2>はじめに</h2>
<p><b>YOLO モザイク処理アプリケーション</b>へようこそ！</p>

<table width="100%" style="background-color: #eff6ff; border: 1px solid #3b82f6;">
<tr><td style="padding: 10px;">
このアプリケーションは、YOLOv11を使用したインスタンスセグメンテーションの全工程をサポートします。
</td></tr>
</table>

<h3>3つの主要機能</h3>
<ul>
<li><b>アノテーション</b> - ポリゴン形式での領域指定</li>
<li><b>学習</b> - YOLOv11モデルの学習</li>
<li><b>推論・モザイク処理</b> - 学習済みモデルを使った検出とモザイク処理</li>
</ul>

<h3>主な特徴</h3>
<table width="100%" style="background-color: #f0fdf4; border: 1px solid #16a34a;">
<tr><td style="padding: 10px;">
<ul style="margin: 0;">
<li>直感的なポリゴン描画によるアノテーション</li>
<li>GPU/CPU自動選択（Windows CUDA、Mac Metal対応）</li>
<li>多様なモザイク処理（黒塗り、白塗り、タイルモザイクなど）</li>
<li>PNGメタデータ保持オプション</li>
<li>バッチ処理による大量画像の一括処理</li>
</ul>
</td></tr>
</table>

<h3>推奨動作環境</h3>
<table width="60%" style="border: 1px solid #e5e7eb;">
<tr style="background-color: #f3f4f6;">
    <td style="padding: 8px; font-weight: bold;">項目</td>
    <td style="padding: 8px; font-weight: bold;">推奨スペック</td>
</tr>
<tr>
    <td style="padding: 8px;">Python</td>
    <td style="padding: 8px;">3.8以上</td>
</tr>
<tr>
    <td style="padding: 8px;">GPU</td>
    <td style="padding: 8px;">NVIDIA RTX / Apple Silicon（推奨）</td>
</tr>
<tr>
    <td style="padding: 8px;">メモリ</td>
    <td style="padding: 8px;">8GB以上</td>
</tr>
</table>
""",
            
            "dataset": style + """
<h2>データセット作成と自動保存</h2>

<h3>データセットの選択</h3>
<ol>
<li>アノテーションタブの「データセット選択」をクリック</li>
<li>既存データセットまたは新規作成を選択</li>
<li>アノテーションする画像フォルダを選択</li>
</ol>

<table width="100%" style="background-color: #eff6ff; border: 1px solid #3b82f6; margin: 10px 0;">
<tr><td style="padding: 10px; font-family: monospace;">
データセット/<br>
├── train/<br>
│   ├── images/    ← 訓練用画像<br>
│   └── labels/    ← 訓練用ラベル<br>
├── valid/<br>
│   ├── images/    ← 検証用画像<br>
│   └── labels/    ← 検証用ラベル<br>
└── data.yaml      ← データセット設定
</td></tr>
</table>

<h3>最近使用したデータセット</h3>
<ol>
<li>ツールバーの「最近のデータセット」をクリック</li>
<li>リストからデータセットを選択</li>
<li>ダブルクリックで素早く開く</li>
</ol>

<h3>自動保存機能</h3>
<table width="100%" style="background-color: #f0fdf4; border: 1px solid #16a34a;">
<tr><td style="padding: 10px;">
<ul style="margin: 0;">
<li><b>アノテーション</b>：画像切り替え時に自動保存</li>
<li><b>ポリゴン完成</b>：ポリゴン確定時に自動保存</li>
<li><b>ウィンドウ設定</b>：位置、サイズ、タブ位置を保持</li>
<li><b>データセット履歴</b>：最近使用したデータセットを記憶</li>
</ul>
</td></tr>
</table>

<h3>データセットの整理</h3>
<p>効率的な学習のために、以下の比率でデータを分割することを推奨します：</p>

<table width="100%" style="background-color: #f9fafb; border: 1px solid #e5e7eb; margin: 10px 0;">
<tr><td style="padding: 10px;">
<ul style="margin: 0;">
<li><b>train/</b> ← 学習用データ（80%）</li>
<li><b>valid/</b> ← 検証用データ（20%）</li>
<li><b>test/</b>  ← テスト用データ（オプション）</li>
</ul>
</td></tr>
</table>

<table width="100%" style="background-color: #fef2f2; border: 1px solid #dc2626;">
<tr><td style="padding: 10px;">
<b>注意：</b>学習用と検証用のデータは重複しないようにしてください
</td></tr>
</table>
""",
            
            "annotation": style + """
<h2>アノテーション機能</h2>

<h3>基本的な使い方</h3>
<ol>
<li><b>データセットの選択</b></li>
</ol>

<table width="100%" style="background-color: #eff6ff; border: 1px solid #3b82f6; margin: 0 0 10px 20px;">
<tr><td style="padding: 10px;">
「データセット選択」ボタンをクリック<br>
既存データセットまたは新規作成を選択<br>
アノテーションする画像フォルダを指定
</td></tr>
</table>

<ol start="2">
<li><b>ポリゴンの描画</b></li>
</ol>

<table width="80%" style="border: 1px solid #e5e7eb; margin: 0 0 10px 20px;">
<tr style="background-color: #f3f4f6;">
    <td style="padding: 8px; font-weight: bold;">操作</td>
    <td style="padding: 8px; font-weight: bold;">動作</td>
</tr>
<tr>
    <td style="padding: 8px; background-color: #f9fafb;">左クリック</td>
    <td style="padding: 8px;">点を追加</td>
</tr>
<tr>
    <td style="padding: 8px; background-color: #f9fafb;">ダブルクリック または Enter</td>
    <td style="padding: 8px;">ポリゴンを確定</td>
</tr>
<tr>
    <td style="padding: 8px; background-color: #f9fafb;">Esc</td>
    <td style="padding: 8px;">現在の描画をキャンセル</td>
</tr>
</table>

<table width="100%" style="background-color: #f0fdf4; border: 1px solid #16a34a; margin: 0 0 10px 20px;">
<tr><td style="padding: 10px;">
<b>ヒント：</b>最低3点でポリゴンを作成できます
</td></tr>
</table>

<ol start="3">
<li><b>クラス（ラベル）の管理</b></li>
</ol>

<table width="100%" style="background-color: #eff6ff; border: 1px solid #3b82f6; margin: 0 0 10px 20px;">
<tr><td style="padding: 10px;">
<b>クラスの表示：</b>データセットに登録されたクラスのみが表示されます<br>
<b>クラス追加：</b>「クラス追加」ボタンから新しいクラスを追加<br>
<b>クラス削除：</b>「クラス削除」ボタンから不要なクラスを削除<br>
<b>初期クラス：</b>新規データセットには「object」クラスが設定されます<br>
<b>色の自動割り当て：</b>各クラスには視覚的に判別しやすい異なる色が自動的に割り当てられます
</td></tr>
</table>

<ol start="4">
<li><b>編集機能</b></li>
</ol>

<table width="80%" style="border: 1px solid #e5e7eb; margin: 0 0 10px 20px;">
<tr style="background-color: #f3f4f6;">
    <td style="padding: 8px; font-weight: bold;">ボタン</td>
    <td style="padding: 8px; font-weight: bold;">機能</td>
</tr>
<tr>
    <td style="padding: 8px;">最後のポリゴンを削除</td>
    <td style="padding: 8px;">直前の操作を取り消し</td>
</tr>
<tr>
    <td style="padding: 8px;">ポリゴンをクリア</td>
    <td style="padding: 8px;">現在の画像の全アノテーションを削除</td>
</tr>
</table>

<h3>自動保存機能</h3>
<div class="warning-box">
<b>重要：</b>別の画像に切り替えると、現在の画像のアノテーションが自動的に保存されます。
</div>

<ul>
<li>保存形式：YOLO形式（<code>.txt</code>）</li>
<li>保存場所：<code>data/annotations/</code></li>
<li>画像と同じファイル名で保存</li>
</ul>

<h3>効率的なアノテーションのコツ</h3>
<div class="tip-box">
<ul>
<li>対象物の輪郭に沿って、適度な間隔で点を配置</li>
<li>複雑な形状では点を多めに、単純な形状では少なめに</li>
<li>キーボードショートカットを活用して作業効率アップ</li>
<li>同じラベルの対象は連続してアノテーション</li>
</ul>
</div>

<h3>アノテーションデータの形式</h3>
<p>YOLO形式で保存されます：</p>
<pre>
クラスID x1 y1 x2 y2 x3 y3 ...
</pre>
<div class="info-box">
座標は画像サイズに対する相対値（0.0～1.0）で保存されます。
</div>
""",
            
            "training": """
<h2>学習機能</h2>

<h3>学習の準備</h3>
<ol>
<li><b>データセットの選択</b>
    <div class="info-box">
    以下の2つの方法でデータセットを選択できます：<br>
    <b>方法1:</b> データセットフォルダをドラッグ＆ドロップ<br>
    <b>方法2:</b> 「データセット選択」ボタンをクリック<br>
    <br>
    選択後、データ分布が自動的に表示されます
    </div>
</li>

<li><b>データセットの構造</b>
    <div class="tip-box">
    train/ - 学習用データ<br>
    ├── images/ - 画像ファイル<br>
    └── labels/ - アノテーションファイル<br>
    valid/ - 検証用データ<br>
    ├── images/ - 画像ファイル<br>
    └── labels/ - アノテーションファイル<br>
    data.yaml - データセット設定ファイル
    </div>
</li>
</ol>

<h3>モデルの選択</h3>
<p>YOLOv11には5つのモデルサイズがあります：</p>
<table>
<tr>
    <th>モデル</th>
    <th>サイズ</th>
    <th>速度</th>
    <th>精度</th>
    <th>推奨用途</th>
</tr>
<tr>
    <td><b>yolo11n-seg</b></td>
    <td>Nano</td>
    <td>高速</td>
    <td>低</td>
    <td>リアルタイム処理</td>
</tr>
<tr>
    <td><b>yolo11s-seg</b></td>
    <td>Small</td>
    <td>速</td>
    <td>低～中</td>
    <td>エッジデバイス</td>
</tr>
<tr>
    <td><b>yolo11m-seg</b></td>
    <td>Medium</td>
    <td>中</td>
    <td>中</td>
    <td><b>一般的な用途（推奨）</b></td>
</tr>
<tr>
    <td><b>yolo11l-seg</b></td>
    <td>Large</td>
    <td>遅</td>
    <td>高</td>
    <td>精度重視</td>
</tr>
<tr>
    <td><b>yolo11x-seg</b></td>
    <td>Extra Large</td>
    <td>最遅</td>
    <td>最高</td>
    <td>研究・開発</td>
</tr>
</table>

<h3>学習パラメータ</h3>
<table>
<tr>
    <th>パラメータ</th>
    <th>説明</th>
    <th>推奨値</th>
</tr>
<tr>
    <td><b>エポック数</b></td>
    <td>学習の繰り返し回数</td>
    <td>50-300</td>
</tr>
<tr>
    <td><b>バッチサイズ</b></td>
    <td>一度に処理する画像数</td>
    <td>8-32（GPUメモリに依存）</td>
</tr>
<tr>
    <td><b>画像サイズ</b></td>
    <td>入力画像のサイズ</td>
    <td>640（標準）</td>
</tr>
<tr>
    <td><b>Early Stopping</b></td>
    <td>改善が見られない場合に自動停止</td>
    <td>50エポック</td>
</tr>
</table>

<h3>学習の実行</h3>
<ol>
<li>パラメータを設定</li>
<li>「学習開始」ボタンをクリック</li>
<li>進行状況がリアルタイムで表示されます</li>
<li>学習済みモデルは<code>data/models/</code>に保存</li>
</ol>

<h3>GPU/CPUの自動選択</h3>
<div class="info-box">
<p>システムが自動的に最適なデバイスを選択：</p>
<ul>
<li><b>Windows：</b>NVIDIA CUDA対応GPU</li>
<li><b>Mac：</b>Apple Silicon (Metal)</li>
<li><b>その他：</b>CPU</li>
</ul>
</div>

<h3>学習のコツ</h3>
<div class="tip-box">
<ul>
<li>1. 最初は小さなモデル（n/s）で試す</li>
<li>2. 過学習を防ぐため、検証データを必ず用意</li>
<li>3. バッチサイズはGPUメモリに合わせて調整</li>
<li>4. 学習曲線を確認して適切なエポック数を決定</li>
</ul>
</div>
""",
            
            "inference": """
<h2>推論・モザイク処理</h2>

<h3>基本的な使い方</h3>
<ol>
<li><b>モデルの選択</b>
    <div class="info-box">
    学習済みモデルがドロップダウンに表示<br>
    <code>best.pt</code> - 最良の性能のモデル<br>
    <code>last.pt</code> - 最後のエポックのモデル
    </div>
</li>

<li><b>推論パラメータ</b>
    <table>
    <tr>
        <th>パラメータ</th>
        <th>説明</th>
        <th>推奨値</th>
    </tr>
    <tr>
        <td><b>信頼度閾値</b></td>
        <td>検出の信頼度（0.0-1.0）</td>
        <td>0.25</td>
    </tr>
    <tr>
        <td><b>IoU閾値</b></td>
        <td>重複除去の閾値（0.0-1.0）</td>
        <td>0.45</td>
    </tr>
    </table>
</li>

<li><b>画像の選択と推論</b>
    <div class="tip-box">
    「画像を選択」 または 「フォルダを選択」<br>
    ↓<br>
    「推論実行」ボタンをクリック<br>
    ↓<br>
    検出結果が画像上に表示
    </div>
</li>
</ol>

<h3>モザイク処理の種類</h3>
<table>
<tr>
    <th>タイプ</th>
    <th>説明</th>
    <th>パラメータ</th>
</tr>
<tr>
    <td><b>gaussian</b></td>
    <td>ガウシアンブラー</td>
    <td>強度（1-50）</td>
</tr>
<tr>
    <td><b>pixelate</b></td>
    <td>ピクセレート</td>
    <td>強度（1-50）</td>
</tr>
<tr>
    <td><b>blur</b></td>
    <td>シンプルブラー</td>
    <td>強度（1-50）</td>
</tr>
<tr>
    <td><b>black</b></td>
    <td>黒塗りつぶし</td>
    <td>なし</td>
</tr>
<tr>
    <td><b>white</b></td>
    <td>白塗りつぶし</td>
    <td>なし</td>
</tr>
<tr>
    <td><b>tile</b></td>
    <td>タイルモザイク</td>
    <td>タイルサイズ（1-100px）</td>
</tr>
</table>

<h3>処理の流れ</h3>
<ol>
<li>推論実行で対象を検出</li>
<li>モザイクタイプとパラメータを選択</li>
<li>「モザイク適用」をクリック</li>
<li>「画像を保存」で結果を保存</li>
</ol>

<h3>PNGメタデータの保持</h3>
<div class="info-box">
<p>PNG画像の場合、メタデータ（EXIF情報など）を保持するオプション：</p>
<ul>
<li>「PNGメタデータを保持」にチェック</li>
<li>画像生成AIのプロンプト情報などを維持</li>
<li>ファイルサイズが若干大きくなる可能性</li>
</ul>
</div>
""",
            
            "dataset_manager": """
<h2>データセット管理</h2>

<h3>データセット管理タブの機能</h3>
<p>データセット管理タブでは、作成したデータセットの一覧表示と管理を行います。</p>

<h3>データセット一覧</h3>
<table>
<tr>
    <th>表示項目</th>
    <th>説明</th>
</tr>
<tr>
    <td><b>データセット名</b></td>
    <td>データセットフォルダの名前</td>
</tr>
<tr>
    <td><b>クラス数</b></td>
    <td>data.yamlに定義されたクラス数</td>
</tr>
<tr>
    <td><b>画像数</b></td>
    <td>全体の画像ファイル数（train + valid）</td>
</tr>
<tr>
    <td><b>最終更新</b></td>
    <td>データセットの最終更新日時</td>
</tr>
</table>

<h3>データセット操作</h3>
<div class="info-box">
<p>以下の操作は右クリックメニューまたはツールバーから実行できます：</p>
</div>

<table>
<tr>
    <th>操作</th>
    <th>説明</th>
</tr>
<tr>
    <td><b>詳細情報</b></td>
    <td>データセットの詳細情報とサンプル画像の表示</td>
</tr>
<tr>
    <td><b>名前変更</b></td>
    <td>データセットフォルダの名前を変更</td>
</tr>
<tr>
    <td><b>削除</b></td>
    <td>データセットを完全に削除（確認ダイアログあり）</td>
</tr>
<tr>
    <td><b>Finderで開く</b></td>
    <td>データセットフォルダをFinderで直接開く</td>
</tr>
</table>

<h3>データセットの詳細情報</h3>
<p>詳細情報ダイアログでは以下を確認できます：</p>
<ul>
<li><b>基本情報：</b>パス、クラス数、クラス名</li>
<li><b>データ分布：</b>訓練/検証データの画像数とラベル数</li>
<li><b>ラベル率：</b>ラベル付けされた画像の割合</li>
<li><b>サンプル画像：</b>ランダムに選択された画像のプレビュー</li>
</ul>

<h3>データセット構造</h3>
<table width="100%" style="background-color: #eff6ff; border: 1px solid #3b82f6; margin: 10px 0;">
<tr><td style="padding: 10px; font-family: monospace;">
datasets/<br>
├── sample_dataset/<br>
│   ├── train/<br>
│   │   ├── images/    ← 訓練用画像<br>
│   │   └── labels/    ← 訓練用ラベル（YOLO形式）<br>
│   ├── valid/<br>
│   │   ├── images/    ← 検証用画像<br>
│   │   └── labels/    ← 検証用ラベル<br>
│   └── data.yaml      ← データセット設定<br>
└── other_dataset/
</td></tr>
</table>

<h3>自動更新機能</h3>
<div class="tip-box">
<p>データセット一覧は5秒ごとに自動更新されます。新しく作成されたデータセットは自動的にリストに追加されます。</p>
</div>
""",
            
            "models": """
<h2>モデル管理</h2>

<h3>モデル管理タブの機能</h3>
<p>モデル管理タブでは、学習済みモデルの一覧表示と管理を行います。</p>

<h3>モデル一覧</h3>
<table>
<tr>
    <th>表示項目</th>
    <th>説明</th>
</tr>
<tr>
    <td><b>モデル名</b></td>
    <td>学習時に自動生成された名前（データセット名_training）</td>
</tr>
<tr>
    <td><b>作成日時</b></td>
    <td>モデルの学習完了日時</td>
</tr>
<tr>
    <td><b>サイズ</b></td>
    <td>モデルフォルダの合計サイズ</td>
</tr>
<tr>
    <td><b>状態</b></td>
    <td>✓ 完了（best.ptが存在）または ⚠ 不完全</td>
</tr>
</table>

<h3>モデル操作</h3>
<div class="info-box">
<p>以下の操作は右クリックメニューまたはツールバーから実行できます：</p>
</div>

<table>
<tr>
    <th>操作</th>
    <th>説明</th>
</tr>
<tr>
    <td><b>詳細情報</b></td>
    <td>学習設定（args.yaml）と学習結果の確認</td>
</tr>
<tr>
    <td><b>名前変更</b></td>
    <td>モデルフォルダの名前を変更</td>
</tr>
<tr>
    <td><b>削除</b></td>
    <td>不要なモデルを完全に削除（確認ダイアログあり）</td>
</tr>
<tr>
    <td><b>Finderで開く</b></td>
    <td>モデルフォルダをエクスプローラー/Finderで直接開く</td>
</tr>
</table>

<h3>モデルの詳細情報</h3>
<p>詳細情報ダイアログでは以下を確認できます：</p>
<ul>
<li><b>基本情報：</b>モデル名、作成日時、最終更新日時</li>
<li><b>学習設定：</b>使用したパラメータ（エポック数、バッチサイズなど）</li>
<li><b>学習結果：</b>最終エポックのmAP、損失値などの指標</li>
</ul>

<h3>自動更新機能</h3>
<div class="tip-box">
<p>モデル一覧は5秒ごとに自動更新されます。学習が完了したモデルは自動的にリストに追加されます。</p>
</div>

<h3>モデルの保存場所</h3>
<table width="100%" style="background-color: #eff6ff; border: 1px solid #3b82f6; margin: 10px 0;">
<tr><td style="padding: 10px; font-family: monospace;">
data/models/<br>
├── sample_dataset_training/<br>
│   ├── train/<br>
│   │   └── weights/<br>
│   │       ├── best.pt    ← 最良モデル<br>
│   │       └── last.pt    ← 最終モデル<br>
│   ├── args.yaml          ← 学習設定<br>
│   └── results.csv        ← 学習結果<br>
└── other_dataset_training/
</td></tr>
</table>

<h3>注意事項</h3>
<div class="warning-box">
<ul>
<li>削除したモデルは復元できません</li>
<li>名前変更時は特殊文字（/, \, :, *, ?, ", <, >, |）は使用できません</li>
<li>学習中のモデルは削除・名前変更できません</li>
</ul>
</div>
""",
            
            "batch": """
<h2><span class="icon">⚡</span>バッチ処理</h2>

<h3><span class="icon">📦</span>フォルダ単位の一括処理</h3>
<div class="info-box">
<p><span class="icon">🎯</span> 複数の画像を一度に処理する機能です。大量の画像に同じモザイク処理を適用できます。</p>
</div>

<h3><span class="icon">🚀</span>使い方</h3>
<ol>
<li><b>準備</b>
    <div class="tip-box">
    <span class="icon">📁</span> 処理したい画像を1つのフォルダにまとめる<br>
    <span class="icon">🖼️</span> 対応形式：<code>PNG</code>、<code>JPG</code>、<code>JPEG</code><br>
    <span class="icon">📂</span> 出力用の空フォルダを用意
    </div>
</li>

<li><b>設定</b>
    <table>
    <tr>
        <th>設定項目</th>
        <th>内容</th>
    </tr>
    <tr>
        <td>モデル選択</td>
        <td>使用する学習済みモデル</td>
    </tr>
    <tr>
        <td>推論パラメータ</td>
        <td>信頼度・IoU閾値</td>
    </tr>
    <tr>
        <td>モザイクタイプ</td>
        <td>6種類から選択</td>
    </tr>
    <tr>
        <td>PNGメタデータ</td>
        <td>保持するかチェック</td>
    </tr>
    </table>
</li>

<li><b>実行</b>
    <div class="info-box">
    <span class="button-style">バッチ処理</span> ボタンをクリック<br>
    ↓<br>
    <span class="icon">📁</span> 入力フォルダを選択<br>
    ↓<br>
    <span class="icon">📂</span> 出力フォルダを選択<br>
    ↓<br>
    <span class="icon">⚙️</span> 処理が開始されます
    </div>
</li>
</ol>

<h3><span class="icon">📊</span>処理中の表示</h3>
<div class="tip-box">
<ul>
<li><span class="icon">📈</span> プログレスバーで進行状況を表示</li>
<li><span class="icon">📋</span> 各画像の処理結果をリアルタイム表示</li>
<li><span class="icon">⚠️</span> エラーが発生した画像はスキップ</li>
<li><span class="icon">✅</span> 処理完了後、結果サマリーを表示</li>
</ul>
</div>

<h3><span class="icon">💾</span>出力ファイル</h3>
<table>
<tr>
    <th>項目</th>
    <th>説明</th>
</tr>
<tr>
    <td>ファイル名</td>
    <td><code>mosaic_元のファイル名</code></td>
</tr>
<tr>
    <td>検出なし画像</td>
    <td>スキップされます</td>
</tr>
<tr>
    <td>保存場所</td>
    <td>指定した出力フォルダ</td>
</tr>
</table>

<h3><span class="icon">💡</span>大量処理のコツ</h3>
<div class="warning-box">
<ul>
<li><span class="icon">💾</span> メモリ使用量に注意（一度に<b>数百枚程度</b>を推奨）</li>
<li><span class="icon">⚡</span> バッチサイズを調整して処理速度を最適化</li>
<li><span class="icon">🧪</span> 処理前にテスト画像で設定を確認</li>
<li><span class="icon">💿</span> 十分な空きディスク容量を確保</li>
</ul>
</div>
""",
            
            "settings": """
<h2><span class="icon">⚙️</span>設定とカスタマイズ</h2>

<h3><span class="icon">📋</span>プロジェクト設定</h3>
<div class="info-box">
<p><span class="icon">📄</span> 各プロジェクトの設定は<code>config.json</code>に保存されます。</p>
</div>

<h3><span class="icon">🏷️</span>ラベルのカスタマイズ</h3>
<p>デフォルトのラベルを変更する場合：</p>
<ol>
<li><code>src/config.py</code>を編集</li>
<li><code>DEFAULT_CONFIG["annotation"]["labels"]</code>を変更</li>
<li>対応する色も<code>colors</code>で設定</li>
</ol>

<h3><span class="icon">🎨</span>デフォルト値の変更</h3>
<pre>
DEFAULT_CONFIG = {
    "annotation": {
        "labels": ["object", "face", "person"],
        "colors": {
            "object": "#FF0000",    <span style="color: #7f8c8d;">← 赤</span>
            "face": "#00FF00",      <span style="color: #7f8c8d;">← 緑</span>
            "person": "#0000FF"     <span style="color: #7f8c8d;">← 青</span>
        }
    },
    "training": {
        "batch_size": 16,
        "epochs": 100,
        "imgsz": 640
    },
    "inference": {
        "confidence": 0.25,
        "iou": 0.45,
        "blur_strength": 15
    }
}
</pre>

<h3><span class="icon">⌨️</span>ショートカットキー</h3>
<table>
<tr>
    <th>キー</th>
    <th>動作</th>
    <th>使用場面</th>
</tr>
<tr>
    <td><kbd>Enter</kbd></td>
    <td>ポリゴンを確定</td>
    <td>アノテーション時</td>
</tr>
<tr>
    <td><kbd>Esc</kbd></td>
    <td>現在の描画をキャンセル</td>
    <td>アノテーション時</td>
</tr>
<tr>
    <td><kbd>Ctrl</kbd>+<kbd>S</kbd></td>
    <td>プロジェクトを保存</td>
    <td>全画面</td>
</tr>
<tr>
    <td><kbd>Ctrl</kbd>+<kbd>O</kbd></td>
    <td>プロジェクトを開く</td>
    <td>全画面</td>
</tr>
</table>

<h3><span class="icon">🚀</span>パフォーマンス設定</h3>
<div class="tip-box">
<ul>
<li><span class="icon">🖥️</span> <b>GPU使用：</b>自動検出されますが、手動で選択も可能</li>
<li><span class="icon">📊</span> <b>バッチサイズ：</b>GPUメモリに応じて調整</li>
<li><span class="icon">📐</span> <b>画像サイズ：</b>精度と速度のトレードオフ</li>
</ul>
</div>
""",
            
            "troubleshooting": """
<h2><span class="icon">🔧</span>トラブルシューティング</h2>

<h3><span class="icon">❓</span>よくある問題と解決方法</h3>

<h4><span class="icon">1️⃣</span> GPUが認識されない</h4>
<div class="warning-box">
<ul>
<li><span class="icon">🪟</span> <b>Windows：</b>CUDA対応のNVIDIAドライバーをインストール</li>
<li><span class="icon">🍎</span> <b>Mac：</b>macOS 12.0以上でApple Siliconを使用</li>
<li><span class="icon">🐍</span> PyTorchのGPU対応版がインストールされているか確認</li>
</ul>
</div>

<h4><span class="icon">2️⃣</span> メモリ不足エラー</h4>
<div class="tip-box">
<ul>
<li><span class="icon">📉</span> バッチサイズを小さくする（16→8→4→1）</li>
<li><span class="icon">🖼️</span> 画像サイズを小さくする（640→320）</li>
<li><span class="icon">📦</span> モデルサイズを小さくする（x→l→m→s→n）</li>
</ul>
</div>

<h4><span class="icon">3️⃣</span> アノテーションが保存されない</h4>
<table>
<tr>
    <th>確認項目</th>
    <th>対処法</th>
</tr>
<tr>
    <td>書き込み権限</td>
    <td>フォルダの権限を確認</td>
</tr>
<tr>
    <td>フォルダ存在</td>
    <td><code>data/annotations/</code>を作成</td>
</tr>
<tr>
    <td>自動保存</td>
    <td>別の画像に切り替えると保存</td>
</tr>
</table>

<h4><span class="icon">4️⃣</span> 学習が進まない</h4>
<ul>
<li><span class="icon">📊</span> 学習率が適切か確認</li>
<li><span class="icon">📁</span> データセットが正しく設定されているか確認</li>
<li><span class="icon">📄</span> data.yamlのパスが正しいか確認</li>
</ul>

<h4><span class="icon">5️⃣</span> モザイク処理が適用されない</h4>
<ul>
<li><span class="icon">🔍</span> 推論で検出されているか確認</li>
<li><span class="icon">📉</span> 信頼度閾値を下げてみる（0.25→0.1）</li>
<li><span class="icon">✅</span> 適切なモデルが選択されているか確認</li>
</ul>

<h3><span class="icon">⚠️</span>エラーメッセージの対処</h3>

<div class="warning-box">
<h4>"No module named 'ultralytics'"</h4>
<pre>pip install ultralytics</pre>
</div>

<div class="warning-box">
<h4>"CUDA out of memory"</h4>
<p>バッチサイズを小さくするか、画像サイズを縮小してください。</p>
<pre>
# バッチサイズを8に変更
batch_size = 8
</pre>
</div>

<div class="warning-box">
<h4>"Invalid annotation format"</h4>
<p>アノテーションファイルがYOLO形式でない可能性があります。</p>
</div>

<h3><span class="icon">📋</span>ログファイル</h3>
<div class="info-box">
<p>詳細なエラー情報は学習時のログに記録されます：</p>
<table>
<tr>
    <th>項目</th>
    <th>内容</th>
</tr>
<tr>
    <td>場所</td>
    <td><code>data/models/プロジェクト名/</code></td>
</tr>
<tr>
    <td>ファイル</td>
    <td><code>results.csv</code>、<code>train_batch*.jpg</code></td>
</tr>
</table>
</div>

<h3><span class="icon">🆘</span>サポート</h3>
<div class="tip-box">
<p>問題が解決しない場合：</p>
<ul>
<li><span class="icon">🐙</span> GitHubのIssuesで報告</li>
<li><span class="icon">📝</span> エラーメッセージの全文を含める</li>
<li><span class="icon">💻</span> 使用環境（OS、Python版、GPU）を記載</li>
</ul>
</div>
"""
        }
        
        return f"""
        <html>
        <head>
            <style type="text/css">
                body {{ 
                    font-family: Arial, sans-serif; 
                    line-height: 1.6; 
                    color: #333;
                    margin: 10px;
                }}
                
                h2 {{ 
                    color: white;
                    background-color: #3498db;
                    padding: 10px;
                    margin: 10px 0;
                }}
                
                h3 {{ 
                    color: #2980b9;
                    border-left: 4px solid #3498db;
                    padding-left: 10px;
                    margin: 15px 0;
                }}
                
                h4 {{ 
                    color: #34495e;
                    background-color: #ecf0f1;
                    padding: 5px 10px;
                    margin: 10px 0;
                }}
                
                table {{ 
                    border: 1px solid #ddd;
                    border-collapse: collapse;
                    width: 100%;
                    margin: 10px 0;
                }}
                
                th {{ 
                    background-color: #3498db;
                    color: white;
                    padding: 8px;
                    text-align: left;
                    border: 1px solid #ddd;
                }}
                
                td {{ 
                    padding: 8px;
                    border: 1px solid #ddd;
                }}
                
                ul, ol {{ 
                    margin-left: 20px;
                    margin-bottom: 10px;
                }}
                
                li {{ 
                    margin-bottom: 5px;
                }}
                
                b {{ 
                    color: #e74c3c;
                    font-weight: bold;
                }}
                
                pre {{ 
                    background-color: #f4f4f4;
                    border: 1px solid #ddd;
                    padding: 10px;
                    overflow-x: auto;
                    font-family: monospace;
                    margin: 10px 0;
                }}
                
                code {{ 
                    background-color: #f4f4f4;
                    padding: 2px 4px;
                    font-family: monospace;
                }}
                
                .tip-box {{
                    background-color: #d4edda;
                    border: 1px solid #c3e6cb;
                    padding: 10px;
                    margin: 10px 0;
                }}
                
                .warning-box {{
                    background-color: #fff3cd;
                    border: 1px solid #ffeeba;
                    padding: 10px;
                    margin: 10px 0;
                }}
                
                .info-box {{
                    background-color: #d1ecf1;
                    border: 1px solid #bee5eb;
                    padding: 10px;
                    margin: 10px 0;
                }}
                
                .button-style {{
                    background-color: #3498db;
                    color: white;
                    padding: 4px 8px;
                    text-decoration: none;
                }}
                
                kbd {{
                    background-color: #666;
                    color: white;
                    padding: 2px 4px;
                    font-family: monospace;
                    font-size: 0.9em;
                }}
                
                p {{
                    margin-bottom: 10px;
                }}
            </style>
        </head>
        <body>
            {contents.get(tag, "<p>コンテンツが見つかりません。</p>")}
        </body>
        </html>
        """