from PySide6.QtWidgets import (QDialog, QVBoxLayout, QTextBrowser, QPushButton,
                               QHBoxLayout, QLabel)
from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices, QFont
from pathlib import Path

class LicenseDialog(QDialog):
    """ライセンス情報を表示するダイアログ"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("ライセンス情報")
        self.setModal(True)
        self.resize(800, 600)
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # タイトル
        title = QLabel("YOLO Mosaic App - ライセンス情報")
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # 重要な通知
        warning_label = QLabel(
            "⚠️ 重要: このアプリケーションはAGPL-3.0ライセンスのUltralytics YOLOを使用しています。\n"
            "ソースコードの公開が必要です。商用利用の場合は商用ライセンスの取得を推奨します。"
        )
        warning_label.setWordWrap(True)
        warning_label.setStyleSheet("""
            QLabel {
                background-color: #FEF3C7;
                color: #92400E;
                padding: 15px;
                border: 1px solid #F59E0B;
                border-radius: 5px;
                font-size: 14px;
                font-weight: bold;
            }
        """)
        layout.addWidget(warning_label)
        
        # ライセンス情報テキスト
        self.text_browser = QTextBrowser()
        self.text_browser.setOpenExternalLinks(False)
        self.text_browser.anchorClicked.connect(self.on_link_clicked)
        
        # HTMLコンテンツを設定
        html_content = self.get_license_html()
        self.text_browser.setHtml(html_content)
        
        layout.addWidget(self.text_browser)
        
        # ボタンレイアウト
        button_layout = QHBoxLayout()
        
        # GitHubリポジトリボタン
        github_btn = QPushButton("📂 ソースコード (GitHub)")
        github_btn.clicked.connect(lambda: self.open_url("https://github.com/kawaiitemachan/yolo-mosaic-app"))
        github_btn.setStyleSheet("""
            QPushButton {
                background-color: #24292E;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #1A1E22;
            }
        """)
        button_layout.addWidget(github_btn)
        
        # Ultralyticsライセンスボタン
        ultralytics_btn = QPushButton("💼 商用ライセンス (Ultralytics)")
        ultralytics_btn.clicked.connect(lambda: self.open_url("https://www.ultralytics.com/license"))
        ultralytics_btn.setStyleSheet("""
            QPushButton {
                background-color: #7C3AED;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #6D28D9;
            }
        """)
        button_layout.addWidget(ultralytics_btn)
        
        button_layout.addStretch()
        
        # 閉じるボタン
        close_btn = QPushButton("閉じる")
        close_btn.clicked.connect(self.accept)
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: #6B7280;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #4B5563;
            }
        """)
        button_layout.addWidget(close_btn)
        
        layout.addLayout(button_layout)
    
    def get_license_html(self):
        """ライセンス情報のHTMLコンテンツを生成"""
        return """
        <html>
        <head>
            <style>
                body {
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
                    line-height: 1.6;
                    color: #333;
                }
                h2 {
                    color: #1F2937;
                    border-bottom: 2px solid #E5E7EB;
                    padding-bottom: 8px;
                    margin-top: 24px;
                }
                h3 {
                    color: #374151;
                    margin-top: 20px;
                }
                .license-box {
                    background-color: #F9FAFB;
                    border: 1px solid #E5E7EB;
                    border-radius: 4px;
                    padding: 12px;
                    margin: 10px 0;
                }
                .agpl-notice {
                    background-color: #FEE2E2;
                    border: 1px solid #FCA5A5;
                    border-radius: 4px;
                    padding: 12px;
                    margin: 10px 0;
                }
                ul {
                    margin: 10px 0;
                    padding-left: 20px;
                }
                a {
                    color: #3B82F6;
                    text-decoration: none;
                }
                a:hover {
                    text-decoration: underline;
                }
                .source-link {
                    font-size: 18px;
                    font-weight: bold;
                    color: #059669;
                    background-color: #D1FAE5;
                    padding: 10px;
                    border-radius: 4px;
                    display: inline-block;
                    margin: 10px 0;
                }
            </style>
        </head>
        <body>
            <h2>ライセンス準拠について</h2>
            
            <div class="agpl-notice">
                <strong>AGPL-3.0ライセンスの要件を満たすため、本アプリケーションの完全なソースコードを公開しています：</strong><br>
                <a href="https://github.com/kawaiitemachan/yolo-mosaic-app" class="source-link">
                    📂 https://github.com/kawaiitemachan/yolo-mosaic-app
                </a>
            </div>
            
            <h2>主要コンポーネントのライセンス</h2>
            
            <h3>1. Ultralytics YOLO</h3>
            <div class="license-box">
                <strong>ライセンス:</strong> AGPL-3.0（商用ライセンスも利用可能）<br>
                <strong>影響:</strong> アプリケーション全体のソースコード公開が必要<br>
                <strong>注意事項:</strong>
                <ul>
                    <li>学習済みモデルにもAGPL-3.0が適用されます</li>
                    <li>SaaS/クラウドサービスとして提供する場合も公開義務があります</li>
                    <li>商用利用には<a href="https://www.ultralytics.com/license">商用ライセンス</a>の取得を推奨</li>
                </ul>
            </div>
            
            <h3>2. PySide6 (Qt for Python)</h3>
            <div class="license-box">
                <strong>ライセンス:</strong> LGPL v3 / GPL v2 / GPL v3<br>
                <strong>商用利用:</strong> 可能（LGPLの要件を満たす場合）<br>
                <strong>要件:</strong>
                <ul>
                    <li>PySide6のソースコードを提供</li>
                    <li>ユーザーがPySide6を差し替え可能にする</li>
                    <li>PySide6を改変した場合は変更を公開</li>
                </ul>
            </div>
            
            <h3>3. その他のライブラリ</h3>
            <div class="license-box">
                <strong>OpenCV-Python:</strong> Apache 2.0 (v4.5.0+) / BSD 3-Clause (v4.4.0以前)<br>
                <strong>NumPy:</strong> BSD 3-Clause<br>
                <strong>Pillow:</strong> MIT-CMU<br>
                <strong>PyTorch:</strong> BSD 3-Clause<br>
                <strong>Matplotlib:</strong> PSFベースのBSDスタイル<br>
                <strong>PyYAML:</strong> MIT<br>
                <br>
                これらはすべて商用利用可能な寛容なライセンスです。
            </div>
            
            <h2>商用利用について</h2>
            
            <p>このアプリケーションを商用目的で使用する場合：</p>
            
            <ol>
                <li><strong>AGPL-3.0の要件を満たす</strong>
                    <ul>
                        <li>ソースコードを公開し続ける</li>
                        <li>改変版も同じライセンスで公開する</li>
                    </ul>
                </li>
                <li><strong>または、Ultralyticsの商用ライセンスを取得</strong>
                    <ul>
                        <li>ソースコード非公開での商用利用が可能</li>
                        <li><a href="https://www.ultralytics.com/license">ライセンス取得はこちら</a></li>
                    </ul>
                </li>
            </ol>
            
            <h2>貢献について</h2>
            
            <p>このプロジェクトへの貢献を歓迎します！</p>
            <ul>
                <li><a href="https://github.com/kawaiitemachan/yolo-mosaic-app/issues">バグ報告・機能要望</a></li>
                <li><a href="https://github.com/kawaiitemachan/yolo-mosaic-app/pulls">プルリクエスト</a></li>
            </ul>
            
            <p><em>最終更新日: 2025年1月8日</em></p>
        </body>
        </html>
        """
    
    def on_link_clicked(self, url):
        """リンクがクリックされたときの処理"""
        self.open_url(url.toString())
    
    def open_url(self, url):
        """URLをデフォルトブラウザで開く"""
        QDesktopServices.openUrl(QUrl(url))