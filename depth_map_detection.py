import numpy as np
import cv2

# def process_depth_image(depth_image, max_distance_mm):

#     mask = np.where(depth_image == 0, 255, 0).astype(np.uint8)

#     depth_inpainted = cv2.inpaint(depth_image.astype(np.float32), mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
#     depth_filtered = np.where((depth_inpainted > 0) & (depth_inpainted <= max_distance_mm), depth_inpainted, 0)
#     depth_8bit = cv2.normalize(depth_filtered, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
#     depth_colormap = cv2.applyColorMap(depth_8bit, cv2.COLORMAP_JET)

#     return depth_colormap, depth_8bit

def process_depth_image(depth_image, max_distance_mm):

    # ตรวจสอบ depth_image หากเป็น 3-channel ให้แปลงเป็น grayscale ก่อน
    if len(depth_image.shape) == 3 and depth_image.shape[2] == 3:
        depth_gray = cv2.cvtColor(depth_image, cv2.COLOR_BGR2GRAY)
    else:
        depth_gray = depth_image.copy()

    # แปลงเป็น float32
    depth_gray = depth_gray.astype(np.float32)

    # สร้าง mask สำหรับ inpainting (จุดที่ไม่มีข้อมูลจะ = 255)
    mask = np.where(depth_gray == 0, 255, 0).astype(np.uint8)

    # Inpainting
    depth_inpainted = cv2.inpaint(depth_gray, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

    # กรองค่าที่เกิน max_distance_mm
    depth_filtered = np.where((depth_inpainted > 0) & (depth_inpainted <= max_distance_mm), depth_inpainted, 0)

    # แปลงเป็นภาพ 8-bit
    depth_8bit = cv2.normalize(depth_filtered, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # สร้าง Color map
    depth_colormap = cv2.applyColorMap(depth_8bit, cv2.COLORMAP_JET)

    return depth_colormap, depth_8bit


def detect_contours(depth_8bit, obj_min_area):
    depth_binary = cv2.morphologyEx(depth_8bit, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    contours, _ = cv2.findContours(depth_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected_contours = []
    for contour in contours:
        if cv2.contourArea(contour) > obj_min_area:
            x, y, w, h = cv2.boundingRect(contour)
            detected_contours.append((x, y, w, h))
    return detected_contours
