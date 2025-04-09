import numpy as np
import cv2

def make_decision1(depth_map, img_width, FOV, safe_distance):
    # แบ่งภาพเป็น 10 ช่อง
    num_sections = 10
    section_width = img_width // num_sections
    min_depth = np.inf
    best_section = 0

    # หาช่องที่มี depth สูงสุด (เช่น ระยะห่างมากที่สุดจากวัตถุ)
    for i in range(num_sections):
        section = depth_map[:, i*section_width:(i+1)*section_width]
        average_depth = np.mean(section)
        if average_depth < min_depth:
            min_depth = average_depth
            best_section = i

    # คำนวณมุมที่ต้องหมุนกล้อง
    x1, x2 = best_section * section_width, (best_section + 1) * section_width
    mid_point = (x1 + x2) / 2
    turn_angle = int(((mid_point - (img_width / 2)) / img_width) * FOV)

    # ตัดสินใจเคลื่อนที่ไปข้างหน้าหรือเลี้ยว
    if min_depth >= safe_distance:
        command_text = "FORWARD"
    else:
        if mid_point < img_width * 0.5:
            command_text = f"TURN RIGHT {abs(turn_angle)} degrees"
        else:
            command_text = f"TURN LEFT {abs(turn_angle)} degrees"

    return command_text, turn_angle