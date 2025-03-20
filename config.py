# config.py
import pyrealsense2 as rs

def setup_realsense(width,height):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)
    pipeline.start(config)
    return pipeline

CLASS_NAMES = ['fixed_obstacle', 'moving_obstacles', 'target']

RGB_FOLDER_PATH =   r"data\recorded_frames20250317_003725\color_no_label"
DEPTH_FOLDER_PATH = r"data\recorded_frames20250317_003725\depth_colormap"

# Depth Map' name Model Test
name = RGB_FOLDER_PATH.split('\\')
DEPTH_MODEL_NAME = name[1].split('recorded_frames')[1]

YOLO_MODEL_PATH = r"model\yolov8n.pt"