import numpy as np
import cv2

def extract_semantic_features(image, mask=None):
    """
    Extract semantic/clinical features from skin lesion images.
    
    Features include:
    - Color statistics (RGB means, stds)
    - Edge density
    - Lesion compactness
    - Asymmetry indicators
    
    Parameters:
    - image: BGR image (OpenCV format)
    - mask: binary mask (optional). If None, uses HSV-based skin detection.
    
    Returns:
    - 1D numpy array of 12 semantic features (float32)
    """
    
    if image is None:
        return np.zeros(12, dtype=np.float32)
    
    # If no mask provided, estimate lesion region
    if mask is None:
        mask = estimate_lesion_mask(image)
    
    features = []
    
    # 1-3: Mean RGB in lesion
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    r_mean = np.mean(rgb[mask > 0, 0]) if np.any(mask) else 0.0
    g_mean = np.mean(rgb[mask > 0, 1]) if np.any(mask) else 0.0
    b_mean = np.mean(rgb[mask > 0, 2]) if np.any(mask) else 0.0
    features.extend([r_mean / 255.0, g_mean / 255.0, b_mean / 255.0])
    
    # 4-6: Std RGB in lesion
    r_std = np.std(rgb[mask > 0, 0]) if np.any(mask) else 0.0
    g_std = np.std(rgb[mask > 0, 1]) if np.any(mask) else 0.0
    b_std = np.std(rgb[mask > 0, 2]) if np.any(mask) else 0.0
    features.extend([r_std / 255.0, g_std / 255.0, b_std / 255.0])
    
    # 7: HSV saturation mean (color intensity)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    s_mean = np.mean(hsv[mask > 0, 1]) if np.any(mask) else 0.0
    features.append(s_mean / 255.0)
    
    # 8: Edge density (Canny edges within lesion)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges[mask > 0]) / (np.sum(mask) + 1e-6)
    features.append(edge_density / 255.0)
    
    # 9: Lesion compactness (perimeter^2 / area)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        compactness = (perimeter ** 2) / (area + 1e-6) if area > 0 else 0.0
        features.append(min(compactness / 1000.0, 1.0))  # normalize
    else:
        features.append(0.0)
    
    # 10: Lesion solidity (area / convex hull area)
    if len(contours) > 0:
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = area / (hull_area + 1e-6) if hull_area > 0 else 0.0
        features.append(solidity)
    else:
        features.append(0.0)
    
    # 11: Lesion size (normalized by image size)
    lesion_area = np.sum(mask) / (mask.shape[0] * mask.shape[1])
    features.append(lesion_area)
    
    # 12: Color variance (total variance across RGB channels)
    color_var = np.var(rgb[mask > 0, :]) if np.any(mask) else 0.0
    features.append(color_var / 10000.0)  # normalize
    
    return np.array(features, dtype=np.float32)


def estimate_lesion_mask(image, hue_range=(0, 180)):
    """
    Estimate lesion region using HSV thresholding.
    Lesions are typically darker, more saturated areas.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Detect skin/lesion: saturation > 30, value > 60
    lower = np.array([0, 30, 60])
    upper = np.array([180, 255, 255])
    
    mask = cv2.inRange(hsv, lower, upper)
    
    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    return mask