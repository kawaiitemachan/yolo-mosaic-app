import cv2
import numpy as np

def expand_mask(mask, expansion_percent):
    """
    マスクを指定された割合で拡張する
    
    Args:
        mask: バイナリマスク (bool型のnumpy配列)
        expansion_percent: 拡張率（%）
    
    Returns:
        拡張されたマスク
    """
    if expansion_percent <= 0:
        return mask
    
    # マスクの輪郭を取得
    mask_uint8 = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 新しいマスクを作成
    expanded_mask = np.zeros_like(mask, dtype=np.uint8)
    
    for contour in contours:
        # 輪郭の面積を計算
        area = cv2.contourArea(contour)
        if area <= 0:
            continue
        
        # 拡張率から膨張サイズを計算
        # 面積の平方根をベースにして、線形的な拡張を行う
        expansion_pixels = int(np.sqrt(area) * expansion_percent / 100)
        expansion_pixels = max(1, expansion_pixels)  # 最低1ピクセルは拡張
        
        # 輪郭を膨張させる
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                         (2 * expansion_pixels + 1, 2 * expansion_pixels + 1))
        
        # 個別の輪郭マスクを作成
        single_mask = np.zeros_like(mask_uint8)
        cv2.drawContours(single_mask, [contour], -1, 255, -1)
        
        # 膨張処理
        dilated = cv2.dilate(single_mask, kernel, iterations=1)
        
        # 拡張マスクに追加
        expanded_mask = cv2.bitwise_or(expanded_mask, dilated)
    
    return expanded_mask.astype(bool)

def apply_mosaic_to_regions(image, detections, blur_type='gaussian', strength=15, mask_expansion=0):
    """
    検出されたマスク領域にモザイクを適用
    
    Args:
        image: 入力画像
        detections: 検出結果のリスト
        blur_type: モザイクのタイプ
        strength: モザイクの強度
        mask_expansion: マスクの拡張率（%）
    """
    result = image.copy()
    
    for detection in detections:
        mask = detection['mask']
        
        # マスクを拡張
        if mask_expansion > 0:
            mask = expand_mask(mask, mask_expansion)
        
        if blur_type == 'gaussian':
            result = apply_gaussian_blur(result, mask, strength)
        elif blur_type == 'pixelate':
            result = apply_pixelate(result, mask, strength)
        elif blur_type == 'blur':
            result = apply_simple_blur(result, mask, strength)
        elif blur_type == 'black':
            result = apply_black_fill(result, mask)
        elif blur_type == 'white':
            result = apply_white_fill(result, mask)
        elif blur_type == 'tile':
            result = apply_tile_mosaic(result, mask, strength)
    
    return result

def apply_gaussian_blur(image, mask, strength):
    """
    ガウシアンブラーを適用
    """
    kernel_size = strength * 2 + 1
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
    result = image.copy()
    result[mask] = blurred[mask]
    
    return result

def apply_pixelate(image, mask, strength):
    """
    ピクセレート（モザイク）効果を適用
    """
    result = image.copy()
    
    y_indices, x_indices = np.where(mask)
    
    if len(y_indices) == 0:
        return result
    
    y_min, y_max = y_indices.min(), y_indices.max()
    x_min, x_max = x_indices.min(), x_indices.max()
    
    roi = image[y_min:y_max+1, x_min:x_max+1]
    
    pixel_size = max(1, strength)
    
    small_h = max(1, (y_max - y_min + 1) // pixel_size)
    small_w = max(1, (x_max - x_min + 1) // pixel_size)
    
    small = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
    
    pixelated = cv2.resize(small, (x_max - x_min + 1, y_max - y_min + 1), 
                          interpolation=cv2.INTER_NEAREST)
    
    roi_mask = mask[y_min:y_max+1, x_min:x_max+1]
    
    result[y_min:y_max+1, x_min:x_max+1][roi_mask] = pixelated[roi_mask]
    
    return result

def apply_simple_blur(image, mask, strength):
    """
    シンプルなブラー効果を適用
    """
    kernel_size = strength * 2 + 1
    blurred = cv2.blur(image, (kernel_size, kernel_size))
    
    result = image.copy()
    result[mask] = blurred[mask]
    
    return result

def apply_black_fill(image, mask):
    """
    マスク領域を黒で塗りつぶし
    """
    result = image.copy()
    result[mask] = [0, 0, 0]
    return result

def apply_white_fill(image, mask):
    """
    マスク領域を白で塗りつぶし
    """
    result = image.copy()
    result[mask] = [255, 255, 255]
    return result

def apply_tile_mosaic(image, mask, tile_size):
    """
    タイルモザイク効果を適用
    tile_size: タイルのピクセルサイズ
    """
    result = image.copy()
    
    y_indices, x_indices = np.where(mask)
    
    if len(y_indices) == 0:
        return result
    
    y_min, y_max = y_indices.min(), y_indices.max()
    x_min, x_max = x_indices.min(), x_indices.max()
    
    tile_size = max(1, tile_size)
    
    for y in range(y_min, y_max + 1, tile_size):
        for x in range(x_min, x_max + 1, tile_size):
            y_end = min(y + tile_size, y_max + 1)
            x_end = min(x + tile_size, x_max + 1)
            
            tile_mask = mask[y:y_end, x:x_end]
            
            if np.any(tile_mask):
                avg_color = image[y:y_end, x:x_end][tile_mask].mean(axis=0)
                result[y:y_end, x:x_end][tile_mask] = avg_color
    
    return result

def batch_process_images(image_paths, model_path, output_dir, 
                        confidence=0.25, iou=0.45, 
                        blur_type='gaussian', strength=15,
                        preserve_png_info=False, mask_expansion=0, use_bbox=False):
    """
    複数の画像をバッチ処理
    """
    from ultralytics import YOLO
    from pathlib import Path
    from PIL import Image as PILImage
    import os
    
    model = YOLO(model_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    results = []
    
    for image_path in image_paths:
        image_path = Path(image_path)
        
        png_info = None
        if preserve_png_info and image_path.suffix.lower() == '.png':
            try:
                pil_img = PILImage.open(str(image_path))
                png_info = pil_img.info
            except:
                png_info = None
        
        image = cv2.imread(str(image_path))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        predictions = model(str(image_path), conf=confidence, iou=iou)
        
        detections = []
        for r in predictions:
            if use_bbox and r.boxes is not None:
                # バウンディングボックスモード
                for i, box in enumerate(r.boxes.xyxy):
                    cls = int(r.boxes.cls[i])
                    conf = float(r.boxes.conf[i])
                    label = model.names[cls]
                    
                    # バウンディングボックスからマスクを作成
                    mask = np.zeros((image.shape[0], image.shape[1]), dtype=bool)
                    x1, y1, x2, y2 = map(int, box)
                    mask[y1:y2, x1:x2] = True
                    
                    detections.append({
                        'mask': mask,
                        'label': label,
                        'confidence': conf
                    })
            elif not use_bbox and r.masks is not None:
                # セグメンテーションモード
                for i, mask in enumerate(r.masks.data):
                    mask_np = mask.cpu().numpy()
                    mask_resized = cv2.resize(mask_np, (image.shape[1], image.shape[0]))
                    
                    cls = int(r.boxes.cls[i])
                    conf = float(r.boxes.conf[i])
                    label = model.names[cls]
                    
                    detections.append({
                        'mask': mask_resized > 0.5,
                        'label': label,
                        'confidence': conf
                    })
        
        if detections:
            processed = apply_mosaic_to_regions(image_rgb, detections, blur_type, strength, mask_expansion)
            
            output_path = output_dir / f"mosaic_{image_path.name}"
            
            if preserve_png_info and png_info and image_path.suffix.lower() == '.png':
                pil_processed = PILImage.fromarray(processed)
                pil_processed.save(str(output_path), pnginfo=png_info)
            else:
                processed_bgr = cv2.cvtColor(processed, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(output_path), processed_bgr)
            
            results.append({
                'input': str(image_path),
                'output': str(output_path),
                'detections': len(detections)
            })
        else:
            results.append({
                'input': str(image_path),
                'output': None,
                'detections': 0
            })
    
    return results