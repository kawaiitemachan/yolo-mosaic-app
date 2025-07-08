from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                               QListWidget, QComboBox, QLabel, QSlider,
                               QSplitter, QFileDialog, QListWidgetItem,
                               QMessageBox, QDialog, QInputDialog, QSizePolicy,
                               QMenu, QDialogButtonBox, QLineEdit, QGroupBox,
                               QScrollArea)
from PySide6.QtCore import Qt, Signal, QPointF, QTimer, QRectF
from PySide6.QtGui import (QPainter, QPen, QColor, QPolygonF, QImage, QPixmap,
                          QCursor, QAction)
from enum import Enum

import cv2
import numpy as np
from pathlib import Path
import yaml

from ..config import IMAGES_DIR, DEFAULT_CONFIG

class CanvasMode(Enum):
    """キャンバスの操作モード"""
    POLYGON = "polygon"  # ポリゴン作成モード
    HAND = "hand"       # ハンドモード（移動専用）

class ImageCanvas(QWidget):
    polygon_completed = Signal(list)
    zoom_changed = Signal(float)  # ズームレベル変更シグナル
    mode_changed = Signal(CanvasMode)  # モード変更シグナル
    
    def __init__(self):
        super().__init__()
        self.image = None
        self.pixmap = None
        self.current_polygon = []
        self.polygons = []
        self.current_label = "object"
        self.drawing = False
        self.scale = 1.0
        self.offset = QPointF(0, 0)
        self.class_colors = {}  # クラスごとの色を動的に管理
        self.setMinimumSize(400, 300)  # 最小サイズを設定
        
        # 操作モード
        self.mode = CanvasMode.POLYGON  # デフォルトはポリゴンモード
        
        # ズーム関連
        self.min_scale = 0.1  # 10%
        self.max_scale = 5.0  # 500%
        self.zoom_step = 1.1  # ズームステップ
        
        # パン（移動）関連
        self.panning = False
        self.pan_start_pos = QPointF()
        self.pan_start_offset = QPointF()
        self.space_pressed = False  # スペースキー押下状態
        
        # サイズポリシーを設定（拡張可能）
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        # キーボードフォーカスを受け取れるようにする
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        
        # 編集モード関連
        self.selected_polygon_index = -1  # 選択されたポリゴンのインデックス
        self.selected_point_index = -1   # 選択された点のインデックス
        self.hovering_polygon_index = -1 # ホバー中のポリゴンのインデックス
        self.hovering_point_index = -1   # ホバー中の点のインデックス
        self.editing_mode = False         # 編集モード
        self.dragging = False             # ドラッグ中かどうか
        self.hover_start_point = False    # 最初の点にホバー中かどうか
        
        # アンドゥ・リドゥ用のスタック
        self.undo_stack = []
        self.redo_stack = []
        
        # マウストラッキングを有効化
        self.setMouseTracking(True)
        
        # 点の検出範囲（ピクセル）
        self.point_detection_radius = 8
        
        # マウスホイールイベントを受け取る
        self.setAttribute(Qt.WidgetAttribute.WA_AcceptTouchEvents, True)
        
    def load_image(self, image_path):
        self.image = QImage(str(image_path))
        self.pixmap = QPixmap.fromImage(self.image)
        self.current_polygon = []
        self.polygons = []
        
        # 編集状態をリセット
        self.selected_polygon_index = -1
        self.selected_point_index = -1
        self.editing_mode = False
        self.dragging = False
        
        # アンドゥ/リドゥスタックをクリア
        self.undo_stack = []
        self.redo_stack = []
        
        # ウィジェットが表示されてから画像をフィットさせる
        if self.isVisible():
            self.fit_image_to_widget()
        else:
            # ウィジェットがまだ表示されていない場合は遅延実行
            QTimer.singleShot(0, self.fit_image_to_widget)
        
        self.update()
        self.zoom_changed.emit(self.scale * 100)
        
    def fit_image_to_widget(self):
        if not self.pixmap or self.pixmap.width() <= 0 or self.pixmap.height() <= 0:
            return
            
        widget_size = self.size()
        if widget_size.width() <= 0 or widget_size.height() <= 0:
            return
            
        image_width = float(self.pixmap.width())
        image_height = float(self.pixmap.height())
        widget_width = float(widget_size.width())
        widget_height = float(widget_size.height())
        
        # 画像の縦横比を計算
        image_aspect_ratio = image_width / image_height
        
        # ウィジェットに収まるようにスケールを計算（縦横比を保持）
        if (widget_width / widget_height) > image_aspect_ratio:
            # ウィジェットの方が横長の場合、高さに合わせる
            self.scale = widget_height / image_height
        else:
            # ウィジェットの方が縦長の場合、幅に合わせる
            self.scale = widget_width / image_width
        
        # スケール後のサイズを計算
        scaled_width = image_width * self.scale
        scaled_height = image_height * self.scale
        
        # 中央に配置するためのオフセットを計算
        self.offset = QPointF(
            (widget_width - scaled_width) / 2.0,
            (widget_height - scaled_height) / 2.0
        )
        
        # ズームレベルを通知
        self.zoom_changed.emit(self.scale * 100)
    
    def resizeEvent(self, event):
        """ウィジェットのサイズが変更されたときに画像を再フィット"""
        super().resizeEvent(event)
        self.fit_image_to_widget()
        self.update()
    
    def showEvent(self, event):
        """ウィジェットが表示されたときに画像を再フィット"""
        super().showEvent(event)
        self.fit_image_to_widget()
        self.update()
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # 背景を描画（画像の周りを暗くする）
        painter.fillRect(self.rect(), QColor(50, 50, 50))
        
        if not self.pixmap:
            return
        
        # 画像を描画
        painter.save()
        painter.translate(self.offset)
        painter.scale(self.scale, self.scale)
        painter.drawPixmap(0, 0, self.pixmap)
        painter.restore()
        
        # クラス色が設定されていない場合はデフォルトを使用
        if not self.class_colors:
            colors = DEFAULT_CONFIG["annotation"]["colors"]
        else:
            colors = self.class_colors
        
        # ポリゴンを描画（画像と同じ変換を適用）
        painter.save()
        painter.translate(self.offset)
        painter.scale(self.scale, self.scale)
        
        # 既存のポリゴンを描画
        for idx, polygon_data in enumerate(self.polygons):
            polygon = polygon_data["points"]
            label = polygon_data["label"]
            color = QColor(colors.get(label, "#FF0000"))
            
            # 選択されているかホバー中かチェック
            is_selected = idx == self.selected_polygon_index
            is_hovering = idx == self.hovering_polygon_index
            
            # 線の太さと不透明度を調整
            pen_width = 3 / self.scale if is_selected else 2 / self.scale
            if is_hovering and not is_selected:
                color.setAlpha(200)
                pen_width = 2.5 / self.scale
            
            pen = QPen(color, pen_width)
            painter.setPen(pen)
            
            # ポリゴンを描画
            if len(polygon) > 1:
                for i in range(len(polygon)):
                    p1 = polygon[i]
                    p2 = polygon[(i + 1) % len(polygon)]
                    painter.drawLine(p1, p2)
                
                # 塗りつぶし（選択時またはホバー時）
                if is_selected or is_hovering:
                    fill_color = QColor(color)
                    fill_color.setAlpha(50 if is_selected else 30)
                    painter.setBrush(fill_color)
                    points = [p for p in polygon]
                    painter.drawPolygon(points)
                    painter.setBrush(Qt.BrushStyle.NoBrush)
                    
            # 頂点を描画
            for pidx, point in enumerate(polygon):
                painter.setBrush(color)
                radius = 4 / self.scale if is_selected else 3 / self.scale
                
                # 選択された点は大きく表示
                if is_selected and pidx == self.selected_point_index:
                    radius = 5 / self.scale
                    painter.setPen(QPen(Qt.GlobalColor.white, 2 / self.scale))
                    painter.drawEllipse(point, radius, radius)
                    painter.setPen(pen)
                else:
                    painter.drawEllipse(point, radius, radius)
        
        # 作成中のポリゴンを描画
        if self.current_polygon:
            color = QColor(colors.get(self.current_label, "#FF0000"))
            pen = QPen(color, 2 / self.scale)
            pen.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(pen)
            
            if len(self.current_polygon) > 1:
                for i in range(len(self.current_polygon) - 1):
                    painter.drawLine(self.current_polygon[i], self.current_polygon[i + 1])
                
                # 最初の点に戻る線をプレビュー（3点以上の場合）
                if len(self.current_polygon) >= 3 and self.hover_start_point:
                    painter.drawLine(self.current_polygon[-1], self.current_polygon[0])
            
            # 頂点を描画
            for idx, point in enumerate(self.current_polygon):
                painter.setBrush(color)
                radius = 3 / self.scale
                
                # 最初の点は特別に表示（ホバー時はさらに大きく）
                if idx == 0:
                    if self.hover_start_point and len(self.current_polygon) >= 3:
                        radius = 5 / self.scale
                        painter.setPen(QPen(Qt.GlobalColor.white, 2 / self.scale))
                        painter.drawEllipse(point, radius, radius)
                        painter.setPen(pen)
                    else:
                        painter.setPen(QPen(color, 3 / self.scale))
                        painter.drawEllipse(point, radius + 1 / self.scale, radius + 1 / self.scale)
                        painter.setPen(pen)
                        painter.drawEllipse(point, radius, radius)
                else:
                    painter.drawEllipse(point, radius, radius)
        
        painter.restore()
    
    def mousePressEvent(self, event):
        if not self.pixmap:
            return
            
        pos = (event.position() - self.offset) / self.scale
        
        # 中ボタンまたはスペース+左クリックまたはハンドモードでパン開始
        if event.button() == Qt.MouseButton.MiddleButton or \
           (event.button() == Qt.MouseButton.LeftButton and self.space_pressed) or \
           (event.button() == Qt.MouseButton.LeftButton and self.mode == CanvasMode.HAND):
            self.panning = True
            self.pan_start_pos = event.position()
            self.pan_start_offset = self.offset
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return
        
        # 画像の範囲内かチェック
        if not (0 <= pos.x() <= self.pixmap.width() and 0 <= pos.y() <= self.pixmap.height()):
            return
        
        # 左クリック（ポリゴンモードのみ）
        if event.button() == Qt.MouseButton.LeftButton and not self.space_pressed and self.mode == CanvasMode.POLYGON:
            # 既存のポリゴンの点をクリックしたかチェック
            clicked_polygon, clicked_point = self.find_point_at_position(pos)
            
            if clicked_polygon >= 0:
                # 既存のポリゴンを選択
                self.selected_polygon_index = clicked_polygon
                self.selected_point_index = clicked_point
                self.editing_mode = True
                self.dragging = True
                self.current_polygon = []  # 作成中のポリゴンをクリア
                self.update()
                return
            
            # 新規ポリゴン作成
            if not self.editing_mode:
                # 最初の点をクリックした場合（ポリゴンを閉じる）
                if len(self.current_polygon) >= 3 and self.is_near_point(pos, self.current_polygon[0]):
                    self.complete_polygon()
                else:
                    # ラベルが選択されているかチェック
                    if not self.current_label or self.current_label == "":
                        # ラベルがない場合、親ウィジェットに通知
                        parent = self.parent()
                        while parent and not hasattr(parent, 'check_and_add_label'):
                            parent = parent.parent()
                        if parent and hasattr(parent, 'check_and_add_label'):
                            if not parent.check_and_add_label():
                                return  # ラベルが追加されなかった場合は処理を中止
                    
                    # 新しい点を追加
                    self.current_polygon.append(pos)
                    self.save_state()  # アンドゥ用に状態を保存
                    self.update()
            else:
                # 編集モードを終了
                self.selected_polygon_index = -1
                self.selected_point_index = -1
                self.editing_mode = False
                self.update()
        
        # 右クリック
        elif event.button() == Qt.MouseButton.RightButton:
            # ポリゴンモードの時のみコンテキストメニューを表示
            if self.mode == CanvasMode.POLYGON:
                self.show_context_menu(event.globalPosition().toPoint())
    
    def mouseMoveEvent(self, event):
        if not self.pixmap:
            return
            
        # パン中の場合
        if self.panning:
            delta = event.position() - self.pan_start_pos
            self.offset = self.pan_start_offset + delta
            self.update()
            return
        
        pos = (event.position() - self.offset) / self.scale
        
        # ドラッグ中の場合
        if self.dragging and self.selected_polygon_index >= 0 and self.selected_point_index >= 0:
            if 0 <= pos.x() <= self.pixmap.width() and 0 <= pos.y() <= self.pixmap.height():
                # 選択された点を移動
                self.polygons[self.selected_polygon_index]["points"][self.selected_point_index] = pos
                self.update()
                return
        
        # ホバー処理
        # 既存のポリゴンのホバーチェック
        old_hovering = self.hovering_polygon_index
        self.hovering_polygon_index, self.hovering_point_index = self.find_point_at_position(pos)
        
        # ポリゴンがない場合はポリゴン全体をチェック
        if self.hovering_polygon_index < 0:
            self.hovering_polygon_index = self.find_polygon_at_position(pos)
        
        # 最初の点へのホバーチェック（ポリゴン作成中）
        old_hover_start = self.hover_start_point
        if len(self.current_polygon) >= 3:
            self.hover_start_point = self.is_near_point(pos, self.current_polygon[0])
        else:
            self.hover_start_point = False
        
        # カーソルを変更
        if self.mode == CanvasMode.HAND:
            # ハンドモードでは常に手のカーソル
            self.setCursor(Qt.CursorShape.OpenHandCursor)
        elif self.space_pressed:
            self.setCursor(Qt.CursorShape.OpenHandCursor)
        elif self.hovering_polygon_index >= 0 or self.hover_start_point:
            self.setCursor(Qt.CursorShape.PointingHandCursor)
        else:
            # ポリゴンモードでは十字カーソル
            self.setCursor(Qt.CursorShape.CrossCursor)
        
        # ホバー状態が変わった場合のみ再描画
        if old_hovering != self.hovering_polygon_index or old_hover_start != self.hover_start_point:
            self.update()
    
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton or \
           (event.button() == Qt.MouseButton.LeftButton and self.panning):
            self.panning = False
            # カーソルを適切に更新
            self.update_cursor()
        elif event.button() == Qt.MouseButton.LeftButton:
            if self.dragging:
                self.dragging = False
                self.save_state()  # 編集後の状態を保存
    
    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Return and len(self.current_polygon) >= 3:
            self.complete_polygon()
        elif event.key() == Qt.Key.Key_Escape:
            if self.current_polygon:
                self.current_polygon = []
                self.update()
            elif self.editing_mode:
                self.selected_polygon_index = -1
                self.selected_point_index = -1
                self.editing_mode = False
                self.update()
        elif event.key() == Qt.Key.Key_Delete or event.key() == Qt.Key.Key_Backspace:
            # 選択されたポリゴンを削除
            if self.selected_polygon_index >= 0:
                self.delete_selected_polygon()
        elif event.key() == Qt.Key.Key_Z and event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            # アンドゥ
            self.undo()
        elif event.key() == Qt.Key.Key_Y and event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            # リドゥ
            self.redo()
        elif event.key() == Qt.Key.Key_Space:
            # スペースキーでパンモード
            self.space_pressed = True
            self.setCursor(Qt.CursorShape.OpenHandCursor)
        elif event.key() == Qt.Key.Key_0 and event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            # Ctrl+0でフィット表示
            self.fit_image_to_widget()
            self.update()
        elif event.key() == Qt.Key.Key_Plus and event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            # Ctrl++でズームイン
            self.zoom_in()
        elif event.key() == Qt.Key.Key_Minus and event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            # Ctrl+-でズームアウト
            self.zoom_out()
        elif event.key() == Qt.Key.Key_H:
            # Hキーでハンドモード
            self.set_mode(CanvasMode.HAND)
        elif event.key() == Qt.Key.Key_P:
            # Pキーでポリゴンモード
            self.set_mode(CanvasMode.POLYGON)
        elif event.key() >= Qt.Key.Key_0 and event.key() <= Qt.Key.Key_9:
            # 数字キーは親ウィジェット（AnnotationWidget）に転送
            parent = self.parent()
            while parent and not isinstance(parent, AnnotationWidget):
                parent = parent.parent()
            if parent:
                parent.keyPressEvent(event)
        else:
            # その他のキーは親ウィジェットで処理
            super().keyPressEvent(event)
    
    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key.Key_Space:
            self.space_pressed = False
            if not self.panning:
                # パン中でなければモードに応じたカーソルに戻す
                self.update_cursor()
        else:
            super().keyReleaseEvent(event)
    
    def wheelEvent(self, event):
        """\u30de\u30a6\u30b9\u30db\u30a4\u30fc\u30eb\u3067\u30ba\u30fc\u30e0"""
        if not self.pixmap:
            return
        
        # Ctrlキーを押しているか、または常にホイールでズーム
        # マウス位置を中心にズーム
        mouse_pos = event.position()
        delta = event.angleDelta().y()
        
        if delta > 0:
            # ズームイン
            self.zoom_to_point(mouse_pos, self.zoom_step)
        else:
            # ズームアウト
            self.zoom_to_point(mouse_pos, 1.0 / self.zoom_step)
    
    def zoom_to_point(self, point, factor):
        """\u6307\u5b9a\u30dd\u30a4\u30f3\u30c8\u3092\u4e2d\u5fc3\u306b\u30ba\u30fc\u30e0"""
        # 新しいスケールを計算
        new_scale = self.scale * factor
        new_scale = max(self.min_scale, min(self.max_scale, new_scale))
        
        if new_scale == self.scale:
            return
        
        # マウス位置を画像座標に変換
        img_pos = (point - self.offset) / self.scale
        
        # スケールを更新
        self.scale = new_scale
        
        # オフセットを調整してマウス位置が同じ画像座標を指すようにする
        self.offset = point - img_pos * self.scale
        
        self.update()
        self.zoom_changed.emit(self.scale * 100)
    
    def zoom_in(self):
        """\u30ba\u30fc\u30e0\u30a4\u30f3"""
        center = QPointF(self.width() / 2, self.height() / 2)
        self.zoom_to_point(center, self.zoom_step)
    
    def zoom_out(self):
        """\u30ba\u30fc\u30e0\u30a2\u30a6\u30c8"""
        center = QPointF(self.width() / 2, self.height() / 2)
        self.zoom_to_point(center, 1.0 / self.zoom_step)
    
    def set_zoom(self, zoom_percentage):
        """\u30ba\u30fc\u30e0\u30ec\u30d9\u30eb\u3092\u76f4\u63a5\u8a2d\u5b9a"""
        new_scale = zoom_percentage / 100.0
        new_scale = max(self.min_scale, min(self.max_scale, new_scale))
        
        if new_scale != self.scale:
            center = QPointF(self.width() / 2, self.height() / 2)
            factor = new_scale / self.scale
            self.zoom_to_point(center, factor)
    
    def set_mode(self, mode):
        """操作モードを設定"""
        if self.mode != mode:
            self.mode = mode
            self.mode_changed.emit(mode)
            
            # モード変更時の処理
            if mode == CanvasMode.HAND:
                # ハンドモードでは作成中のポリゴンをキャンセル
                if self.current_polygon:
                    self.current_polygon = []
                    self.update()
                # 編集モードも終了
                if self.editing_mode:
                    self.selected_polygon_index = -1
                    self.selected_point_index = -1
                    self.editing_mode = False
                    self.update()
            
            # カーソルを更新
            self.update_cursor()
    
    def update_cursor(self):
        """現在の状態に応じてカーソルを更新"""
        if self.mode == CanvasMode.HAND:
            self.setCursor(Qt.CursorShape.OpenHandCursor)
        elif self.space_pressed:
            self.setCursor(Qt.CursorShape.OpenHandCursor)
        elif self.hovering_polygon_index >= 0 or self.hover_start_point:
            self.setCursor(Qt.CursorShape.PointingHandCursor)
        else:
            self.setCursor(Qt.CursorShape.CrossCursor)
    
    def complete_polygon(self):
        if len(self.current_polygon) >= 3:
            polygon_data = {
                "points": self.current_polygon.copy(),
                "label": self.current_label
            }
            self.polygons.append(polygon_data)
            self.polygon_completed.emit(self.current_polygon)
            self.current_polygon = []
            self.update()
    
    def set_label(self, label):
        self.current_label = label
    
    def set_class_colors(self, classes):
        """クラスごとの色を設定"""
        # 既存のデフォルト色
        default_colors = DEFAULT_CONFIG["annotation"]["colors"]
        
        # 視覚的に判別しやすい色のリスト（色相環上で離れた色を選択）
        distinctive_colors = [
            # 基本色
            "#FF0000",  # 赤
            "#0000FF",  # 青
            "#00FF00",  # 緑
            "#FFFF00",  # 黄色
            "#FF00FF",  # マゼンタ
            "#00FFFF",  # シアン
            "#FF8800",  # オレンジ
            "#8800FF",  # 紫
            
            # 明度を変えた基本色
            "#CC0000",  # 暗い赤
            "#0000CC",  # 暗い青
            "#00CC00",  # 暗い緑
            "#CCCC00",  # 暗い黄色
            "#CC00CC",  # 暗いマゼンタ
            "#00CCCC",  # 暗いシアン
            "#CC6600",  # 暗いオレンジ
            "#6600CC",  # 暗い紫
            
            # パステルカラー（背景から目立つように調整）
            "#FF6666",  # ライトレッド
            "#6666FF",  # ライトブルー
            "#66FF66",  # ライトグリーン
            "#FFFF66",  # ライトイエロー
            "#FF66FF",  # ライトマゼンタ
            "#66FFFF",  # ライトシアン
            "#FFB366",  # ライトオレンジ
            "#B366FF",  # ライトパープル
        ]
        
        self.class_colors = {}
        used_colors = set()
        
        # まず、デフォルトの色を使用
        for class_name in classes:
            if class_name in default_colors:
                self.class_colors[class_name] = default_colors[class_name]
                used_colors.add(default_colors[class_name])
        
        # 残りのクラスに対して、使用されていない色から割り当て
        available_colors = [c for c in distinctive_colors if c not in used_colors]
        color_index = 0
        
        for class_name in classes:
            if class_name not in self.class_colors:
                if color_index < len(available_colors):
                    self.class_colors[class_name] = available_colors[color_index]
                else:
                    # 全ての色を使い切った場合は、最初から再利用
                    self.class_colors[class_name] = available_colors[color_index % len(available_colors)]
                color_index += 1
    
    def clear_polygons(self):
        self.polygons = []
        self.current_polygon = []
        self.update()
    
    def undo_last_polygon(self):
        if self.polygons:
            self.polygons.pop()
            self.update()
    
    def find_point_at_position(self, pos):
        """指定位置にある点を検索して、ポリゴンインデックスと点インデックスを返す"""
        for poly_idx, polygon_data in enumerate(self.polygons):
            for point_idx, point in enumerate(polygon_data["points"]):
                if self.is_near_point(pos, point):
                    return poly_idx, point_idx
        return -1, -1
    
    def find_polygon_at_position(self, pos):
        """指定位置にあるポリゴンを検索"""
        for idx, polygon_data in enumerate(self.polygons):
            if self.point_in_polygon(pos, polygon_data["points"]):
                return idx
        return -1
    
    def is_near_point(self, pos1, pos2):
        """2つの点が近いかどうかを判定"""
        distance = ((pos1.x() - pos2.x()) ** 2 + (pos1.y() - pos2.y()) ** 2) ** 0.5
        return distance <= self.point_detection_radius / self.scale
    
    def point_in_polygon(self, point, polygon):
        """点がポリゴン内にあるかどうかを判定（レイキャスティング法）"""
        if len(polygon) < 3:
            return False
        
        x, y = point.x(), point.y()
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0].x(), polygon[0].y()
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n].x(), polygon[i % n].y()
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
    
    def show_context_menu(self, global_pos):
        """コンテキストメニューを表示"""
        menu = QMenu(self)
        
        # 現在のマウス位置を画像座標に変換
        local_pos = self.mapFromGlobal(global_pos)
        pos = (QPointF(local_pos) - self.offset) / self.scale
        
        # ポリゴンまたは点を右クリックした場合
        poly_idx, point_idx = self.find_point_at_position(pos)
        if poly_idx < 0:
            poly_idx = self.find_polygon_at_position(pos)
        
        if poly_idx >= 0:
            delete_action = menu.addAction("ポリゴンを削除")
            delete_action.triggered.connect(lambda: self.delete_polygon(poly_idx))
            
            menu.addSeparator()
            
            # ラベル変更サブメニュー
            label_menu = menu.addMenu("ラベルを変更")
            for label in self.class_colors.keys():
                if label != self.polygons[poly_idx]["label"]:
                    action = label_menu.addAction(label)
                    action.triggered.connect(lambda checked, l=label, idx=poly_idx: self.change_polygon_label(idx, l))
        
        # 一般的なアクション
        if menu.actions():
            menu.addSeparator()
        
        if len(self.current_polygon) >= 3:
            complete_action = menu.addAction("ポリゴンを確定")
            complete_action.triggered.connect(self.complete_polygon)
        
        if self.current_polygon:
            cancel_action = menu.addAction("作成をキャンセル")
            cancel_action.triggered.connect(lambda: self.cancel_current_polygon())
        
        if self.undo_stack:
            undo_action = menu.addAction("元に戻す (Ctrl+Z)")
            undo_action.triggered.connect(self.undo)
        
        if self.redo_stack:
            redo_action = menu.addAction("やり直す (Ctrl+Y)")
            redo_action.triggered.connect(self.redo)
        
        if menu.actions():
            menu.exec(global_pos)
    
    def delete_polygon(self, index):
        """指定インデックスのポリゴンを削除"""
        if 0 <= index < len(self.polygons):
            self.save_state()
            self.polygons.pop(index)
            self.selected_polygon_index = -1
            self.selected_point_index = -1
            self.editing_mode = False
            self.update()
    
    def delete_selected_polygon(self):
        """選択されたポリゴンを削除"""
        if self.selected_polygon_index >= 0:
            self.delete_polygon(self.selected_polygon_index)
    
    def change_polygon_label(self, index, new_label):
        """ポリゴンのラベルを変更"""
        if 0 <= index < len(self.polygons):
            self.save_state()
            self.polygons[index]["label"] = new_label
            self.update()
    
    def cancel_current_polygon(self):
        """現在作成中のポリゴンをキャンセル"""
        self.current_polygon = []
        self.update()
    
    def save_state(self):
        """現在の状態をアンドゥスタックに保存"""
        # 現在のポリゴンの深いコピーを作成
        state = {
            "polygons": [{
                "points": [QPointF(p) for p in poly["points"]],
                "label": poly["label"]
            } for poly in self.polygons],
            "current_polygon": [QPointF(p) for p in self.current_polygon]
        }
        self.undo_stack.append(state)
        
        # アンドゥスタックが大きくなりすぎないように制限
        if len(self.undo_stack) > 50:
            self.undo_stack.pop(0)
        
        # 新しい操作が行われたらリドゥスタックをクリア
        self.redo_stack.clear()
    
    def undo(self):
        """元に戻す"""
        if len(self.undo_stack) > 1:
            # 現在の状態をリドゥスタックに保存
            current_state = self.undo_stack.pop()
            self.redo_stack.append(current_state)
            
            # 前の状態を復元
            prev_state = self.undo_stack[-1]
            self.polygons = [{
                "points": [QPointF(p) for p in poly["points"]],
                "label": poly["label"]
            } for poly in prev_state["polygons"]]
            self.current_polygon = [QPointF(p) for p in prev_state["current_polygon"]]
            
            self.selected_polygon_index = -1
            self.selected_point_index = -1
            self.editing_mode = False
            self.update()
    
    def redo(self):
        """やり直す"""
        if self.redo_stack:
            # 現在の状態をアンドゥスタックに保存
            self.save_state()
            self.undo_stack.pop()  # save_stateで追加された重複を削除
            
            # リドゥスタックから状態を復元
            state = self.redo_stack.pop()
            self.polygons = [{
                "points": [QPointF(p) for p in poly["points"]],
                "label": poly["label"]
            } for poly in state["polygons"]]
            self.current_polygon = [QPointF(p) for p in state["current_polygon"]]
            
            self.selected_polygon_index = -1
            self.selected_point_index = -1
            self.editing_mode = False
            self.update()

class AnnotationWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.dataset_path = None
        self.images_folder = None
        self.current_split = "train"  # train or valid
        self.current_image_index = -1  # 現在選択されている画像のインデックス
        self.init_ui()
        self.current_image_path = None
        
        # フォーカスポリシーを設定してキーボードイベントを受け取れるようにする
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
    
    def closeEvent(self, event):
        """ウィジェットが閉じられる時に自動保存"""
        if self.current_image_path and self.canvas.polygons:
            self.save_annotations(silent=True)
        event.accept()
        
    def init_ui(self):
        layout = QHBoxLayout(self)
        
        self.canvas = ImageCanvas()
        self.canvas.polygon_completed.connect(self.on_polygon_completed)
        self.canvas.zoom_changed.connect(self.on_zoom_changed)
        self.canvas.mode_changed.connect(self.on_mode_changed)
        left_panel = self.create_left_panel()
        
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(self.canvas)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        
        layout.addWidget(splitter)
    
    def create_left_panel(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # データセット選択ボタン
        select_dataset_btn = QPushButton("データセット選択")
        select_dataset_btn.clicked.connect(self.select_dataset)
        layout.addWidget(select_dataset_btn)
        
        # データセット情報表示
        self.dataset_info_label = QLabel("データセット未選択")
        self.dataset_info_label.setWordWrap(True)
        layout.addWidget(self.dataset_info_label)
        
        # train/valid切り替え
        layout.addWidget(QLabel("データ分割:"))
        self.split_combo = QComboBox()
        self.split_combo.addItems(["train (訓練用)", "valid (検証用)"])
        self.split_combo.currentIndexChanged.connect(self.on_split_changed)
        layout.addWidget(self.split_combo)
        
        # ズームコントロール
        zoom_group = QGroupBox("ズーム")
        zoom_layout = QVBoxLayout(zoom_group)
        
        # ズームレベル表示
        self.zoom_label = QLabel("100%")
        self.zoom_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.zoom_label.setStyleSheet("QLabel { font-size: 14px; font-weight: bold; }")
        zoom_layout.addWidget(self.zoom_label)
        
        # ズームボタン
        zoom_btn_layout = QHBoxLayout()
        self.zoom_in_btn = QPushButton("+")
        self.zoom_in_btn.clicked.connect(self.canvas.zoom_in)
        self.zoom_in_btn.setToolTip("ズームイン (Ctrl++, ホイール上)")
        self.zoom_out_btn = QPushButton("-")
        self.zoom_out_btn.clicked.connect(self.canvas.zoom_out)
        self.zoom_out_btn.setToolTip("ズームアウト (Ctrl+-, ホイール下)")
        self.zoom_fit_btn = QPushButton("フィット")
        self.zoom_fit_btn.clicked.connect(self.fit_image)
        self.zoom_fit_btn.setToolTip("画像をウィンドウにフィット (Ctrl+0)")
        
        zoom_btn_layout.addWidget(self.zoom_out_btn)
        zoom_btn_layout.addWidget(self.zoom_fit_btn)
        zoom_btn_layout.addWidget(self.zoom_in_btn)
        zoom_layout.addLayout(zoom_btn_layout)
        
        # ズームスライダー
        self.zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self.zoom_slider.setMinimum(10)  # 10%
        self.zoom_slider.setMaximum(500)  # 500%
        self.zoom_slider.setValue(100)  # 100%
        self.zoom_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.zoom_slider.setTickInterval(100)
        self.zoom_slider.valueChanged.connect(self.on_zoom_slider_changed)
        zoom_layout.addWidget(self.zoom_slider)
        
        # プリセットズームボタン
        preset_layout = QHBoxLayout()
        for percentage in [50, 100, 200]:
            btn = QPushButton(f"{percentage}%")
            btn.clicked.connect(lambda checked, p=percentage: self.canvas.set_zoom(p))
            preset_layout.addWidget(btn)
        zoom_layout.addLayout(preset_layout)
        
        layout.addWidget(zoom_group)
        
        # モード選択
        mode_group = QGroupBox("操作モード")
        mode_layout = QVBoxLayout(mode_group)
        
        # 現在のモード表示
        self.mode_label = QLabel("ポリゴン作成モード")
        self.mode_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.mode_label.setStyleSheet("""
            QLabel {
                font-size: 14px;
                font-weight: bold;
                padding: 8px;
                background-color: #3b82f6;
                color: white;
                border-radius: 5px;
            }
        """)
        mode_layout.addWidget(self.mode_label)
        
        # モード切り替えボタン
        mode_btn_layout = QHBoxLayout()
        
        self.polygon_mode_btn = QPushButton("ポリゴン [P]")
        self.polygon_mode_btn.clicked.connect(lambda: self.canvas.set_mode(CanvasMode.POLYGON))
        self.polygon_mode_btn.setCheckable(True)
        self.polygon_mode_btn.setChecked(True)
        self.polygon_mode_btn.setToolTip("ポリゴン作成モード (Pキー)")
        
        self.hand_mode_btn = QPushButton("ハンド [H]")
        self.hand_mode_btn.clicked.connect(lambda: self.canvas.set_mode(CanvasMode.HAND))
        self.hand_mode_btn.setCheckable(True)
        self.hand_mode_btn.setToolTip("ハンドモード - 画像移動専用 (Hキー)")
        
        # ボタングループ風のスタイル
        button_style = """
            QPushButton {
                padding: 8px;
                border: 1px solid #d1d5db;
                background-color: white;
            }
            QPushButton:checked {
                background-color: #3b82f6;
                color: white;
                border: 1px solid #3b82f6;
            }
            QPushButton:hover {
                background-color: #f3f4f6;
            }
            QPushButton:checked:hover {
                background-color: #2563eb;
            }
        """
        self.polygon_mode_btn.setStyleSheet(button_style)
        self.hand_mode_btn.setStyleSheet(button_style)
        
        mode_btn_layout.addWidget(self.polygon_mode_btn)
        mode_btn_layout.addWidget(self.hand_mode_btn)
        mode_layout.addLayout(mode_btn_layout)
        
        layout.addWidget(mode_group)
        
        layout.addWidget(QLabel("画像リスト:"))
        self.image_list = QListWidget()
        self.image_list.itemClicked.connect(self.on_image_selected)
        layout.addWidget(self.image_list)
        
        # 画像ナビゲーションボタン
        nav_layout = QHBoxLayout()
        self.prev_btn = QPushButton("◀ 前の画像")
        self.prev_btn.clicked.connect(self.go_to_previous_image)
        self.next_btn = QPushButton("次の画像 ▶")
        self.next_btn.clicked.connect(self.go_to_next_image)
        nav_layout.addWidget(self.prev_btn)
        nav_layout.addWidget(self.next_btn)
        layout.addLayout(nav_layout)
        
        # ボタンの初期状態を無効に
        self.prev_btn.setEnabled(False)
        self.next_btn.setEnabled(False)
        
        layout.addWidget(QLabel("ラベル:"))
        self.label_combo = QComboBox()
        self.label_combo.currentIndexChanged.connect(self.on_label_changed)
        layout.addWidget(self.label_combo)
        
        # クラス管理ボタン
        class_management_layout = QHBoxLayout()
        add_class_btn = QPushButton("クラス追加")
        add_class_btn.clicked.connect(self.add_class)
        remove_class_btn = QPushButton("クラス削除")
        remove_class_btn.clicked.connect(self.remove_class)
        class_management_layout.addWidget(add_class_btn)
        class_management_layout.addWidget(remove_class_btn)
        layout.addLayout(class_management_layout)
        
        self.clear_btn = QPushButton("ポリゴンをクリア")
        self.clear_btn.clicked.connect(self.canvas.clear_polygons)
        layout.addWidget(self.clear_btn)
        
        self.undo_btn = QPushButton("最後のポリゴンを削除")
        self.undo_btn.clicked.connect(self.canvas.undo_last_polygon)
        layout.addWidget(self.undo_btn)
        
        save_btn = QPushButton("アノテーションを保存")
        save_btn.clicked.connect(self.save_annotations)
        layout.addWidget(save_btn)
        
        # 操作方法をスクロール可能にする
        layout.addWidget(QLabel("操作方法:"))
        
        # スクロールエリアを作成
        help_scroll = QScrollArea()
        help_scroll.setWidgetResizable(True)
        help_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        help_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        
        # 操作説明の内容
        help_text = QLabel(
            """【ポリゴン作成】
• 左クリック: 点を追加
• 最初の点をクリック/Enter: 確定
• Esc: 作成をキャンセル

【ポリゴン編集】
• 既存の点をドラッグ: 移動
• ポリゴンをクリック: 選択
• Delete/Backspace: 削除
• 右クリック: メニュー表示

【ラベル選択】
• 1-9キー: 対応するラベルを選択
• 0キー: 10番目のラベルを選択

【モード切り替え】
• Pキー: ポリゴン作成モード
• Hキー: ハンドモード

【ズーム・パン】
• ホイール: ズームイン/アウト
• 中ボタンドラッグ: 画像を移動
• スペース+ドラッグ: 画像を移動
• ハンドモード: 左ドラッグで移動
• Ctrl+0: フィット表示
• Ctrl++/-: ズームイン/アウト

【画像ナビゲーション】
• ←/→ キー: 前後の画像へ移動
• ボタンクリック: 前後の画像へ

【その他】
• Ctrl+Z: 元に戻す
• Ctrl+Y: やり直す"""
        )
        help_text.setWordWrap(True)
        help_text.setStyleSheet("QLabel { background-color: #f0f0f0; padding: 10px; border-radius: 3px; }")
        
        # スクロールエリアに設定
        help_scroll.setWidget(help_text)
        help_scroll.setStyleSheet("""
            QScrollArea {
                background-color: #f0f0f0;
                border: 1px solid #ddd;
                border-radius: 3px;
            }
            QScrollBar:vertical {
                width: 12px;
            }
        """)
        
        # 最大高さを設定してコンパクトに
        help_scroll.setMaximumHeight(300)
        
        layout.addWidget(help_scroll)
        
        layout.addStretch()
        return widget
    
    def on_label_changed(self, index):
        """ラベル選択が変更されたときの処理"""
        if index >= 0 and index < self.label_combo.count():
            # 実際のクラス名を取得
            class_name = self.label_combo.itemData(index)
            if class_name:
                self.canvas.set_label(class_name)
    
    def select_dataset(self):
        """データセット選択ダイアログを表示"""
        from .dataset_dialog import DatasetSelectionDialog
        
        dialog = DatasetSelectionDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.dataset_path = dialog.dataset_path
            self.images_folder = dialog.images_path
            
            # データセット情報を更新
            self.update_dataset_info()
            
            # 画像リストを更新
            self.load_images_from_folder()
            
            # 自動保存通知
            self.notify_auto_save()
    
    def load_dataset(self, dataset_path):
        """指定されたデータセットを読み込む"""
        self.dataset_path = dataset_path
        
        # データセットから画像フォルダを推測
        # train/imagesがあればそれを使用
        train_images = dataset_path / "train" / "images"
        if train_images.exists():
            self.images_folder = train_images
        else:
            # データセット内の画像を探す
            for img_dir in dataset_path.rglob("images"):
                if img_dir.is_dir():
                    self.images_folder = img_dir
                    break
        
        # データセット情報を更新
        self.update_dataset_info()
        
        # 画像リストを更新
        if self.images_folder:
            self.load_images_from_folder()
        
        # 自動保存通知
        self.notify_auto_save()
    
    def notify_auto_save(self):
        """自動保存の通知を送る"""
        parent = self.parent()
        while parent and not hasattr(parent, 'show_save_notification'):
            parent = parent.parent()
        
        if parent and hasattr(parent, 'show_save_notification'):
            parent.show_save_notification("データセット設定を自動保存しました")
    
    def update_dataset_info(self):
        """データセット情報表示を更新"""
        if self.dataset_path:
            info_text = f"データセット: {self.dataset_path.name}\n"
            
            # data.yamlから情報を読み込む
            yaml_path = self.dataset_path / "data.yaml"
            if yaml_path.exists():
                try:
                    with open(yaml_path, 'r') as f:
                        data = yaml.safe_load(f)
                    classes = data.get('names', [])
                    
                    if classes:
                        info_text += f"クラス: {', '.join(classes)}"
                    else:
                        info_text += "クラス: なし（アノテーション開始時に追加されます）"
                    
                    # ラベルコンボボックスを更新
                    self.update_label_combo(classes)
                except:
                    pass
            
            self.dataset_info_label.setText(info_text)
    
    def on_split_changed(self, index):
        """train/valid切り替え時の処理"""
        self.current_split = "train" if index == 0 else "valid"
        
        # 現在の画像のアノテーションを保存
        if self.current_image_path and self.canvas.polygons:
            self.save_annotations(silent=True)
    
    def load_images_from_folder(self):
        """指定されたフォルダから画像を読み込む"""
        if not self.images_folder:
            return
            
        self.image_list.clear()
        self.current_image_index = -1
        
        for img_path in sorted(self.images_folder.glob("*.jpg")) + sorted(self.images_folder.glob("*.png")):
            item = QListWidgetItem(img_path.name)
            item.setData(Qt.ItemDataRole.UserRole, str(img_path))
            self.image_list.addItem(item)
        
        # ナビゲーションボタンの状態を更新
        self.update_navigation_buttons()
    
    def on_image_selected(self, item):
        # 現在の画像のアノテーションを自動保存
        if self.current_image_path and self.canvas.polygons:
            self.save_annotations(silent=True)
        
        # インデックスを更新
        self.current_image_index = self.image_list.row(item)
        
        image_path = item.data(Qt.ItemDataRole.UserRole)
        self.current_image_path = Path(image_path)
        self.canvas.load_image(image_path)
        
        # ズームコントロールをリセット
        self.zoom_slider.setValue(100)
        
        # 既存のアノテーションを読み込む
        self.load_existing_annotations()
        
        # ナビゲーションボタンの状態を更新
        self.update_navigation_buttons()
    
    def save_annotations(self, silent=False):
        if not self.current_image_path or not self.canvas.polygons:
            return
        
        if not self.dataset_path:
            if not silent:
                QMessageBox.warning(self, "警告", "データセットが選択されていません")
            return
            
        # データセット構造に従って保存
        from ..annotation.yolo_format import save_yolo_segmentation_to_dataset
        
        try:
            txt_path = save_yolo_segmentation_to_dataset(
                self.current_image_path,
                self.canvas.polygons,
                self.canvas.pixmap.width(),
                self.canvas.pixmap.height(),
                self.dataset_path,
                self.current_split
            )
            
            # data.yamlを更新
            self.update_data_yaml()
            
            if not silent:
                QMessageBox.information(self, "保存完了", f"アノテーションを保存しました:\n{txt_path}")
        except Exception as e:
            if not silent:
                QMessageBox.critical(self, "エラー", f"保存中にエラーが発生しました: {str(e)}")
    
    def load_existing_annotations(self):
        """既存のアノテーションファイルを読み込む"""
        if not self.current_image_path or not self.dataset_path:
            return
            
        from ..annotation.yolo_format import load_yolo_segmentation_from_dataset
        
        try:
            polygons = load_yolo_segmentation_from_dataset(
                self.current_image_path,
                self.dataset_path,
                self.current_split,
                self.canvas.pixmap.width(),
                self.canvas.pixmap.height()
            )
            self.canvas.polygons = polygons
            # 初期状態をアンドゥスタックに保存
            self.canvas.save_state()
            self.canvas.update()
        except:
            # アノテーションファイルが存在しない場合は無視
            # 空の状態を初期状態として保存
            self.canvas.save_state()
    
    def update_data_yaml(self):
        """data.yamlのクラス情報を更新"""
        if not self.dataset_path:
            return
            
        yaml_path = self.dataset_path / "data.yaml"
        
        try:
            # 既存のdata.yamlを読み込む
            with open(yaml_path, 'r') as f:
                data = yaml.safe_load(f)
            
            # 現在のクラス情報を取得（ここでは更新しない）
            # クラスの追加・削除はadd_class/remove_classメソッドで行う
            
        except Exception as e:
            print(f"data.yaml読み込みエラー: {e}")
    
    def on_polygon_completed(self, polygon):
        """ポリゴンが完成したときの処理"""
        # ポリゴン完成時に自動保存
        if hasattr(self, 'dataset_path') and self.dataset_path:
            self.save_annotations(silent=True)
    
    def update_label_combo(self, classes):
        """ラベルコンボボックスを更新"""
        current_label = self.label_combo.currentText()
        self.label_combo.clear()
        
        if classes:
            # ショートカットキー付きでラベルを表示
            for i, class_name in enumerate(classes):
                if i < 9:
                    display_text = f"[{i+1}] {class_name}"
                elif i == 9:
                    display_text = f"[0] {class_name}"
                else:
                    display_text = f"     {class_name}"
                self.label_combo.addItem(display_text)
                # 実際のクラス名を保存
                self.label_combo.setItemData(i, class_name)
            
            # Canvasの色設定を更新
            self.canvas.set_class_colors(classes)
            
            # 前の選択を復元
            current_index = -1
            for i in range(self.label_combo.count()):
                if self.label_combo.itemData(i) == current_label:
                    current_index = i
                    break
            
            if current_index >= 0:
                self.label_combo.setCurrentIndex(current_index)
            elif classes:
                self.label_combo.setCurrentIndex(0)
                self.canvas.set_label(classes[0])
        else:
            # ラベルがない場合、Canvasにも空のラベルを設定
            self.canvas.set_label("")
    
    def add_class(self):
        """クラスを追加"""
        if not self.dataset_path:
            QMessageBox.warning(self, "警告", "先にデータセットを選択してください")
            return
        
        # クラス名入力ダイアログ
        class_name, ok = QInputDialog.getText(
            self, 
            "クラス追加", 
            "新しいクラス名を入力してください:",
            text=""
        )
        
        if ok and class_name:
            # 既存のクラスを取得
            yaml_path = self.dataset_path / "data.yaml"
            if yaml_path.exists():
                try:
                    with open(yaml_path, 'r') as f:
                        data = yaml.safe_load(f)
                    
                    classes = data.get('names', [])
                    
                    # 重複チェック
                    if class_name in classes:
                        QMessageBox.warning(self, "警告", f"クラス '{class_name}' は既に存在します")
                        return
                    
                    # クラスを追加
                    classes.append(class_name)
                    data['names'] = classes
                    data['nc'] = len(classes)
                    
                    # 保存
                    with open(yaml_path, 'w') as f:
                        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
                    
                    # UI更新
                    self.update_dataset_info()
                    self.notify_auto_save()
                    
                except Exception as e:
                    QMessageBox.critical(self, "エラー", f"クラス追加エラー: {str(e)}")
    
    def remove_class(self):
        """クラスを削除"""
        if not self.dataset_path:
            QMessageBox.warning(self, "警告", "先にデータセットを選択してください")
            return
        
        # 現在のクラスを取得
        yaml_path = self.dataset_path / "data.yaml"
        if not yaml_path.exists():
            return
        
        try:
            with open(yaml_path, 'r') as f:
                data = yaml.safe_load(f)
            
            classes = data.get('names', [])
            
            if not classes:
                QMessageBox.information(self, "情報", "削除するクラスがありません")
                return
            
            # クラス選択ダイアログ
            class_name, ok = QInputDialog.getItem(
                self,
                "クラス削除",
                "削除するクラスを選択してください:",
                classes,
                0,
                False
            )
            
            if ok and class_name:
                # 確認ダイアログ
                reply = QMessageBox.question(
                    self,
                    "確認",
                    f"クラス '{class_name}' を削除しますか？\n注意: このクラスのアノテーションは手動で更新する必要があります。",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                
                if reply == QMessageBox.StandardButton.Yes:
                    # クラスを削除
                    classes.remove(class_name)
                    data['names'] = classes
                    data['nc'] = len(classes)
                    
                    # 保存
                    with open(yaml_path, 'w') as f:
                        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
                    
                    # UI更新
                    self.update_dataset_info()
                    self.notify_auto_save()
                    
        except Exception as e:
            QMessageBox.critical(self, "エラー", f"クラス削除エラー: {str(e)}")
    
    def go_to_previous_image(self):
        """前の画像に移動"""
        if self.current_image_index > 0:
            self.image_list.setCurrentRow(self.current_image_index - 1)
            item = self.image_list.item(self.current_image_index - 1)
            if item:
                self.on_image_selected(item)
    
    def go_to_next_image(self):
        """次の画像に移動"""
        if self.current_image_index < self.image_list.count() - 1:
            self.image_list.setCurrentRow(self.current_image_index + 1)
            item = self.image_list.item(self.current_image_index + 1)
            if item:
                self.on_image_selected(item)
    
    def update_navigation_buttons(self):
        """ナビゲーションボタンの有効/無効状態を更新"""
        count = self.image_list.count()
        
        # 前の画像ボタン
        self.prev_btn.setEnabled(self.current_image_index > 0)
        
        # 次の画像ボタン
        self.next_btn.setEnabled(self.current_image_index >= 0 and self.current_image_index < count - 1)
        
        # 現在の位置を表示（オプション）
        if self.current_image_index >= 0 and count > 0:
            status_text = f"画像 {self.current_image_index + 1} / {count}"
            # ステータスバーがあれば更新
            parent = self.parent()
            while parent:
                if hasattr(parent, 'status_bar'):
                    parent.status_bar.showMessage(status_text)
                    break
                parent = parent.parent()
    
    def keyPressEvent(self, event):
        """キーボードショートカットの処理"""
        if event.key() == Qt.Key.Key_Left:
            # 左矢印キー: 前の画像へ
            self.go_to_previous_image()
        elif event.key() == Qt.Key.Key_Right:
            # 右矢印キー: 次の画像へ
            self.go_to_next_image()
        elif event.key() >= Qt.Key.Key_1 and event.key() <= Qt.Key.Key_9:
            # 数字キー1-9: ラベル選択（ポリゴンモードのみ）
            if self.canvas.mode == CanvasMode.POLYGON:
                index = event.key() - Qt.Key.Key_1  # 0-8のインデックスに変換
                if index < self.label_combo.count():
                    self.label_combo.setCurrentIndex(index)
        elif event.key() == Qt.Key.Key_0:
            # 数字キー0: 10番目のラベル選択（ポリゴンモードのみ）
            if self.canvas.mode == CanvasMode.POLYGON and self.label_combo.count() >= 10:
                self.label_combo.setCurrentIndex(9)
        elif event.key() == Qt.Key.Key_H:
            # Hキー: ハンドモード
            self.canvas.set_mode(CanvasMode.HAND)
        elif event.key() == Qt.Key.Key_P:
            # Pキー: ポリゴンモード
            self.canvas.set_mode(CanvasMode.POLYGON)
        else:
            # その他のキーはデフォルト処理
            super().keyPressEvent(event)
    
    def check_and_add_label(self):
        """ラベルが存在しない場合、ユーザーに追加を促す"""
        if self.label_combo.count() == 0:
            # ラベルがない場合、ダイアログを表示
            dialog = QDialog(self)
            dialog.setWindowTitle("ラベルの追加")
            dialog.setModal(True)
            dialog.resize(400, 200)
            
            layout = QVBoxLayout(dialog)
            
            # メッセージ
            message = QLabel(
                "アノテーションを開始するにはラベルが必要です。\n"
                "新しいラベルを追加してください。"
            )
            message.setWordWrap(True)
            layout.addWidget(message)
            
            # ラベル入力フィールド
            input_layout = QHBoxLayout()
            input_layout.addWidget(QLabel("ラベル名:"))
            label_input = QLineEdit()
            label_input.setPlaceholderText("例: person, car, object など")
            input_layout.addWidget(label_input)
            layout.addLayout(input_layout)
            
            # ボタン
            button_box = QDialogButtonBox(
                QDialogButtonBox.StandardButton.Ok | 
                QDialogButtonBox.StandardButton.Cancel
            )
            button_box.accepted.connect(dialog.accept)
            button_box.rejected.connect(dialog.reject)
            layout.addWidget(button_box)
            
            # ダイアログを表示
            if dialog.exec() == QDialog.DialogCode.Accepted:
                new_label = label_input.text().strip()
                if new_label:
                    # ラベルを追加
                    self.add_class_with_name(new_label)
                    return True
                else:
                    QMessageBox.warning(
                        self,
                        "警告",
                        "ラベル名を入力してください。"
                    )
                    return False
            else:
                return False
        
        return True  # ラベルが既に存在する場合
    
    def add_class_with_name(self, class_name):
        """指定された名前でクラスを追加"""
        if not self.dataset_path:
            return
            
        yaml_path = self.dataset_path / "data.yaml"
        
        try:
            # data.yamlを読み込む
            with open(yaml_path, 'r') as f:
                data = yaml.safe_load(f)
            
            classes = data.get('names', [])
            
            # 既に存在する場合はスキップ
            if class_name in classes:
                return
                
            # クラスを追加
            classes.append(class_name)
            data['names'] = classes
            data['nc'] = len(classes)
            
            # 保存
            with open(yaml_path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
            
            # UI更新
            self.update_dataset_info()
            self.notify_auto_save()
            
            # 新しく追加したラベルを選択
            for i in range(self.label_combo.count()):
                if self.label_combo.itemData(i) == class_name:
                    self.label_combo.setCurrentIndex(i)
                    break
            
        except Exception as e:
            QMessageBox.critical(self, "エラー", f"クラス追加エラー: {str(e)}")
    
    def on_zoom_changed(self, zoom_percentage):
        """ズームレベルが変更されたときの処理"""
        self.zoom_label.setText(f"{int(zoom_percentage)}%")
        
        # スライダーを更新（valueChangedシグナルを一時的に無効化）
        self.zoom_slider.blockSignals(True)
        self.zoom_slider.setValue(int(zoom_percentage))
        self.zoom_slider.blockSignals(False)
    
    def on_zoom_slider_changed(self, value):
        """ズームスライダーが変更されたときの処理"""
        self.canvas.set_zoom(value)
    
    def fit_image(self):
        """画像をウィンドウにフィット"""
        self.canvas.fit_image_to_widget()
        self.canvas.update()
    
    def on_mode_changed(self, mode):
        """モードが変更されたときの処理"""
        if mode == CanvasMode.POLYGON:
            self.mode_label.setText("ポリゴン作成モード")
            self.mode_label.setStyleSheet("""
                QLabel {
                    font-size: 14px;
                    font-weight: bold;
                    padding: 8px;
                    background-color: #3b82f6;
                    color: white;
                    border-radius: 5px;
                }
            """)
            self.polygon_mode_btn.setChecked(True)
            self.hand_mode_btn.setChecked(False)
            
            # ラベル選択やポリゴン操作ボタンを有効化
            self.label_combo.setEnabled(True)
            self.clear_btn.setEnabled(True)
            self.undo_btn.setEnabled(True)
            
        else:  # CanvasMode.HAND
            self.mode_label.setText("ハンドモード")
            self.mode_label.setStyleSheet("""
                QLabel {
                    font-size: 14px;
                    font-weight: bold;
                    padding: 8px;
                    background-color: #16a34a;
                    color: white;
                    border-radius: 5px;
                }
            """)
            self.hand_mode_btn.setChecked(True)
            self.polygon_mode_btn.setChecked(False)
            
            # ハンドモードではポリゴン操作を無効化
            self.label_combo.setEnabled(False)
            self.clear_btn.setEnabled(False)
            self.undo_btn.setEnabled(False)