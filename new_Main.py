import cv2
import os
import torch
import numpy as np
from datetime import datetime

import pyrealsense2 as rs
from ultralytics import YOLO

from new_record_frame import setup_frame_savers, save_frame
from new_record_video import setup_video_writers
from new_yolo_detection import load_yolo_model, detect_objects
from new_depth_map_detection import process_depth_image, detect_contours
from new_combine_detection import merge_detections
from new_path_decision import make_decision
from new_path_decision1 import make_decision1
from new_get_3d_coordinates import get_3D_coordinates_and_rgb
from new_plan_path_with_3d_point_cloud import detect_obstacles_and_plan_path

def main():
    model_path = "yolov8n.pt"

    width = 640
    height = 480
    FOV = 48
    if width * height < 600000:
        FOV = 48
    else:
        FOV = 64
    
    bag_file = "data\output_20250323_165929-001.bag"

    pipeline = rs.pipeline()
    config = rs.config()
    
    # IMPORTANT: Enable reading from .bag file
    config.enable_device_from_file(bag_file, repeat_playback=False)

    MAX_DISTANCE_M = 5.0  # Maximum distance to consider (in meter)
    MAX_DISTANCE_MM = int(MAX_DISTANCE_M * 1000)  # Convert meters to mm
    CONF_THRESHOLD = 0.05  # Minimum confidence for YOLO detections
    OBJ_MIN_AREA = (500 * width * height) / (640 * 480)  # Minimum area for depth-based contour detection
    SAFE_DISTANCE = 1  # Distance in mm to be considered safe for moving forward
    ANGLE_STABILIZATION = 5  # Prevent angle fluctuation by allowing minor variations
    prev_turn_angle = 0  # Initialize the previous turn angle

    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)  # RGB Color Camera

# เริ่มต้นการใช้งาน
    
    profile = pipeline.start(config)
    #config.enable_stream(rs.stream.depth, width, height, rs.format.z16, 30)
    #config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)
    x_bounds = (-0.15, 0.15)
    y_bounds = (-0.15, 0.15)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    record = 0 # Set 0 for no recorded, 1 for reccorded frames
    realtime = 0  # Set to 0 for real-time processing, 1 for processing frames, 2 for processing videos
    bag_filename = "output_" + timestamp + ".bag"  # สร้างชื่อไฟล์แบบ timestamp อัตโนมัติ

    if record == 2 and realtime == 0:
        recorded_frame = "recorded_frames" + str(timestamp)
        rgb_out, depth_out, rgb_no_label_out = setup_frame_savers(recorded_frame)
        config.enable_record_to_file(bag_filename)
    elif record == 1 and realtime == 0:
        rgb_out, depth_out, rgb_no_label_out = setup_video_writers("recorded_videos" + str(timestamp))
        config.enable_record_to_file(bag_filename)
    else:
        pass

    model = load_yolo_model(model_path)
    
    try:
        if realtime == 0:
            loop = iter(lambda: pipeline.wait_for_frames(), None)

        elif realtime == 1:
            depth_folder_path = "D:/EE Engineer/Year4/project/main/recorded_frames20250317_092415/depth_colormap"  # Path to the folder containing saved depth frames
            color_folder_path = "D:/EE Engineer/Year4/project/main/recorded_frames20250317_092415/color_no_label"   # Path to the folder containing saved color frames
            depth_frame_files = sorted(os.listdir(depth_folder_path))
            color_frame_files = sorted(os.listdir(color_folder_path))
            loop = zip(depth_frame_files, color_frame_files)
        
        for frames in loop:
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            color_no_label = color_image.copy()  # Copy for no-label video
            depth_sensor = profile.get_device().first_depth_sensor()
            depth_scale = depth_sensor.get_depth_scale()
            depth_intrinsics = rs.video_stream_profile(profile.get_stream(rs.stream.depth)).get_intrinsics()
            
            fov_x = 2 * np.degrees(np.arctan(depth_intrinsics.width / (2 * depth_intrinsics.fx)))
            fov_y = 2 * np.degrees(np.arctan(depth_intrinsics.height / (2 * depth_intrinsics.fy)))
            x = np.linspace(0, depth_intrinsics.width - 1, depth_intrinsics.width)
            y = np.linspace(0, depth_intrinsics.height - 1, depth_intrinsics.height)
            
            if depth_image is None or color_image is None:
                continue
                
            depth_colormap, depth_8bit,inpainted_depth,depth_display = process_depth_image(depth_image, MAX_DISTANCE_MM)
            detected_contours = detect_contours(depth_8bit, OBJ_MIN_AREA)
            detected_objects = detect_objects(model, color_image, CONF_THRESHOLD, depth_image)
   
            grid_x, grid_y = np.meshgrid(x, y)
            depth = inpainted_depth[grid_y.astype(np.int32), grid_x.astype(np.int32)] * depth_scale
            X = (grid_x - depth_intrinsics.ppx) / depth_intrinsics.fx * depth
            Y = (grid_y - depth_intrinsics.ppy) / depth_intrinsics.fy * depth
            Z = depth

            valid = depth > 0
            X = X[valid]
            Y = Y[valid]
            Z = Z[valid]
            merge_detections(color_image, depth_colormap, detected_contours, detected_objects, depth_image, model)
            #command_text, prev_turn_angle = make_decision1(depth_image, width, FOV, SAFE_DISTANCE)
            #command_text, prev_turn_angle = make_decision(detected_objects, SAFE_DISTANCE, ANGLE_STABILIZATION, prev_turn_angle, width, FOV)
                        
            command_text, prev_turn_angle = detect_obstacles_and_plan_path(X, Y, Z, fov_x, fov_y, x_bounds, y_bounds, SAFE_DISTANCE,ANGLE_STABILIZATION,prev_turn_angle)
            
            cv2.putText(color_image, command_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            #while command_text != "FORWARD":
                
            if record == 1:
                rgb_out.write(color_image)
                depth_out.write(depth_colormap)
                rgb_no_label_out.write(color_no_label)
            else:
                pass

            cv2.imshow('RGB Camera (Obstacle Detection)', color_image)
            cv2.imshow('Depth Map', depth_display)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        if realtime == 0:
            pipeline.stop()
        if record in [1, 2]:
            rgb_out.release()
            depth_out.release()
            rgb_no_label_out.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
