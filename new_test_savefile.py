import pyrealsense2 as rs
import numpy as np
import cv2 as cv2

# กำหนดชื่อไฟล์ bag
bag_file = "output_20250323_170057.bag"

pipeline = rs.pipeline()
config = rs.config()

# ใช้ไฟล์ .bag แทนการดึงข้อมูลแบบ real-time
config.enable_device_from_file(bag_file, repeat_playback=False)

# เปิดสตรีม depth และ color
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # RGB Color Camera

# เริ่มต้นการใช้งาน
pipeline.start(config)

# ค่าระยะสูงสุดที่ต้องการแสดงผล (5 เมตร)
MAX_DISTANCE_M = 3.0 #5 เมตร
MAX_DISTANCE_MM = int(MAX_DISTANCE_M * 1000)  # แปลงเป็นมิลลิเมตร (5000 มม.)

# กำหนดพารามิเตอร์สำหรับการกรองพื้น
GROUND_THRESHOLD = 300  # ถ้าค่าความลึกเปลี่ยนแปลง < 300 mm → ถือว่าเป็นพื้น
OBJ_MIN_AREA = 500  # พื้นที่ขั้นต่ำที่วัตถุต้องมี เพื่อไม่ให้ Noise เล็ก ๆ ถูกตรวจจับ

try:
    while True:
        # อ่านข้อมูลจากกล้อง
        frames = pipeline.wait_for_frames()
        
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()  # RGB Camera
        
        if not depth_frame or not color_frame:
            continue
        #print("frame:",depth_frame)
        # แปลง Depth Frame เป็น Numpy array
        depth_image = np.asanyarray(depth_frame.get_data())
        #print("image:",depth_image)
        # กรองเฉพาะระยะ ≤ 5 เมตร (5000 มม.)
        depth_filtered = np.where((depth_image > 0) & (depth_image <= 2000), depth_image, 0)

        # ** กรองพื้นออกโดยใช้ Gradient Filtering **
        depth_gradient = cv2.Laplacian(depth_filtered.astype(np.float32), cv2.CV_32F)
        depth_filtered[np.abs(depth_gradient) < GROUND_THRESHOLD] = 0  # ลบพื้น

        # แปลง Depth Map ให้เป็น 8-bit Grayscale
        depth_8bit = cv2.convertScaleAbs(depth_filtered, alpha=0.03)
        depth_colormap = cv2.applyColorMap(depth_8bit, cv2.COLORMAP_JET)

        # ** หาวัตถุโดยใช้ Contour Detection **
        _, depth_binary = cv2.threshold(depth_8bit, 1, 255, cv2.THRESH_BINARY)
        depth_binary = cv2.morphologyEx(depth_binary, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))  # Closing operation

        # หา Contours ที่เป็นสิ่งกีดขวาง
        contours, _ = cv2.findContours(depth_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # แปลง Color Frame เป็น Numpy array (BGR)
        color_image = np.asanyarray(color_frame.get_data())  #

        detected_objects = []

        for contour in contours:
            if cv2.contourArea(contour) > OBJ_MIN_AREA:  # กรอง Noise ออก
                x, y, w, h = cv2.boundingRect(contour)
                center_x, center_y = x + w // 2, y + h // 2

                # ห ิ    าค่าระยะของ Bounding Box
                obj_depth = np.median(depth_filtered[y:y+h, x:x+w][depth_filtered[y:y+h, x:x+w] > 0])

                if obj_depth > 0:
                    detected_objects.append((center_x, center_y, obj_depth))

                    # วาด Bounding Box และระบุระยะวัตถุ
                    cv2.rectangle(color_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    text = f"{obj_depth / 1000:.2f} m"
                    cv2.putText(color_image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # แสดงผลภาพ
        cv2.imshow('Depth Map (Filtered, ≤ 5m)', depth_colormap)
        cv2.imshow('RGB Camera (Obstacle Detection)', color_image)

        # กด 'q' เพื่อออกจากโปรแกรม
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
