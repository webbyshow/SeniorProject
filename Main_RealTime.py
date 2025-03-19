# Main_RealTime.py
import pyrealsense2 as rs
import numpy as np
from datetime import datetime
import torch
import os
import cv2
from ultralytics import YOLO

from record_frame import setup_frame_savers , save_frame
from config import setup_realsense
from record_video import setup_video_writers
from yolo_detection import load_yolo_model, detect_objects
from depth_map_detection import process_depth_image, detect_contours
from combine_detection import merge_detections  
#from path_decision_Histogram_of_Depths_new import make_decision
from path_decision import make_decision

def main():
    model_path = "yolo_v8n.pt" #### open
    width = 640
    height = 480
    FOV = 48
    if width * height < 600000:
        FOV = 48
    else:
        FOV = 64 #### close
    MAX_DISTANCE_M = 5.0  # Maximum distance to consider (In meter)
    MAX_DISTANCE_MM = int(MAX_DISTANCE_M * 1000)  # Convert meters to mm
    CONF_THRESHOLD = 0.05  # Minimum confidence for YOLO detections
    OBJ_MIN_AREA = (500 * width * height) / (640 * 480)  # Minimum area for depth-based contour detection
    SAFE_DISTANCE = 1500  # Distance in mm to be considered safe for moving forward
    ANGLE_STABILIZATION = 5  # Prevent angle fluctuation by allowing minor variations
 
    frame_index = 0 #### open
    prev_turn_angle = None  # Initialize the previous turn angle #### close

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)
    pipeline.start(config)

    #rgb_out, depth_out, rgb_no_label_out = setup_video_writers("recorded_videos")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") #### Open
    print(timestamp)
    recorded_frame = "recorded_frames" + str(timestamp)
    rgb_out, depth_out, rgb_no_label_out = setup_frame_savers(recorded_frame)
    model = load_yolo_model(model_path) #### Close

    try:
        while True:
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            depth_image = np.asanyarray(depth_frame.get_data())     #### Open
            color_image = np.asanyarray(color_frame.get_data())
            color_no_label = color_image.copy()                     #### Close

            depth_colormap, depth_8bit = process_depth_image(depth_image, MAX_DISTANCE_MM)
            detected_contours = detect_contours(depth_8bit, OBJ_MIN_AREA)
            detected_objects = detect_objects(model, color_image, CONF_THRESHOLD,depth_image)

            merge_detections(color_image, depth_colormap, detected_objects, depth_image, model)

            #ลองหา make_decision2,3,4 มาเปลี่ยนได้
            command_text, prev_turn_angle = make_decision(detected_objects, SAFE_DISTANCE, ANGLE_STABILIZATION, prev_turn_angle, width,FOV)
            #command_text, prev_turn_angle = make_decision(width, depth_image, SAFE_DISTANCE, FOV, prev_turn_angle, ANGLE_STABILIZATION)
            
            #command_text, prev_turn_angle = make_decision(width, depth_image, SAFE_DISTANCE, FOV)
            cv2.putText(color_image, command_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            save_frame(color_image, recorded_frame + "/color_image", frame_index)
            save_frame(depth_colormap,  recorded_frame + "/depth_colormap", frame_index)
            save_frame(color_no_label,  recorded_frame + "/color_no_label", frame_index)
            frame_index += 1 

            cv2.imshow('RGB Camera (Obstacle Detection)', color_image)
            cv2.imshow('Depth Map', depth_colormap)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        pipeline.stop()
        rgb_out.release()
        depth_out.release()
        rgb_no_label_out.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
