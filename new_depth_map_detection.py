import numpy as np
import cv2

def process_depth_image(depth_image, max_distance_mm):
    mask = (depth_image == 0).astype(np.uint8) * 255
    inpainted_depth = cv2.inpaint(depth_image, mask, 3, cv2.INPAINT_NS)
    depth_display = cv2.normalize(inpainted_depth, None, 0, 255, cv2.NORM_MINMAX)
    depth_display = np.uint8(depth_display)
    depth_display = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)
    depth_filtered = np.where((inpainted_depth > 0) & (inpainted_depth <= max_distance_mm), inpainted_depth, 0)
    depth_8bit = cv2.convertScaleAbs(depth_filtered, alpha=0.03)
    depth_colormap = cv2.applyColorMap(depth_8bit, cv2.COLORMAP_JET)
    return depth_colormap, depth_8bit,inpainted_depth,depth_display

def detect_contours(depth_8bit, obj_min_area):
    depth_binary = cv2.morphologyEx(depth_8bit, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    contours, _ = cv2.findContours(depth_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected_contours = []
    for contour in contours:
        if cv2.contourArea(contour) > obj_min_area:
            x, y, w, h = cv2.boundingRect(contour)
            detected_contours.append((x, y, w, h))
    return detected_contours


