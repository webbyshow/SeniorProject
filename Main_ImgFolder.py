# Main_ImgFolder.py
import os
import cv2
import numpy as np
from datetime import datetime
from ultralytics import YOLO

import config
from combine_detection   import merge_detections                                # Script file
from record_frame        import setup_frame_savers,  save_frame                 # Script file
from yolo_detection      import load_yolo_model,     detect_objects             # Script file
from depth_map_detection import process_depth_image, detect_contours            # Script file
from path_decision       import make_decision,       make_depth_based_decision  # Script file

print('Initialize The Project')

def process_image_folder():

    rgb_folder   = config.RGB_FOLDER_PATH
    depth_folder = config.DEPTH_FOLDER_PATH
    model_path   = config.YOLO_MODEL_PATH

    width = 640
    height = 480
    FOV = 48 if width * height < 600000 else 64
    MAX_DISTANCE_M = 5.0                                # Max distance in meters
    MAX_DISTANCE_MM = int(MAX_DISTANCE_M * 1000)        # Convert to mm
    CONF_THRESHOLD = 0.05                               # YOLO confidence threshold
    OBJ_MIN_AREA = (500 * width * height) / (640 * 480) # Minimum contour area
    SAFE_DISTANCE = 1500                                # Safe distance in mm
    ANGLE_STABILIZATION = 5                             # Stabilization factor

    frame_index = 0
    prev_turn_angle = None  # Initialize previous turn angle

    model = load_yolo_model(model_path)

    rgb_files = sorted([f for f in os.listdir(rgb_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])
    depth_files = sorted([f for f in os.listdir(depth_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])

    if len(rgb_files) != len(depth_files):
        print("Error: RGB and Depth folder must have the same number of images!")
        return

    depth_map_model = 'A' ## 'A'/ 'B' / 'C' / 'D'

    ## Set up frame saving folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # recorded_frame_name = "recorded_frames_" + str(timestamp)
    recorded_frame_name = "_depth_" + config.DEPTH_MODEL_NAME + depth_map_model  
    os.makedirs(recorded_frame_name, exist_ok=True)
    
    rgb_out, depth_out, rgb_no_label_out = setup_frame_savers(recorded_frame_name)

    for i, (rgb_file, depth_file) in enumerate(zip(rgb_files, depth_files)):
        rgb_path = os.path.join(rgb_folder, rgb_file)
        depth_path = os.path.join(depth_folder, depth_file)

        color_image = cv2.imread(rgb_path)
        depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)  # Load as grayscale

        if color_image is None or depth_image is None:
            print(f"Skipping frame {i}: Unable to load images.")
            continue

        color_no_label = color_image.copy()

        depth_colormap, depth_8bit = process_depth_image(depth_image, MAX_DISTANCE_MM,depth_map_model) 

        # detected_contours = detect_contours(depth_8bit, OBJ_MIN_AREA)
        detected_objects = detect_objects(model, color_image, CONF_THRESHOLD, depth_image)
        
        command_text, prev_turn_angle = make_decision(detected_objects, SAFE_DISTANCE, ANGLE_STABILIZATION, prev_turn_angle, width, FOV)
        
        if len(detected_objects) == 0 :
            command_text = make_depth_based_decision(depth_image, MAX_DISTANCE_MM, SAFE_DISTANCE, width, FOV)
        else : 
            pass
            
        merge_detections(color_image, depth_colormap, detected_objects, depth_image, model) 

        cv2.putText(color_image, command_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        save_frame(color_image,    recorded_frame_name + "/color_image", frame_index)
        save_frame(depth_colormap, recorded_frame_name + "/depth_colormap", frame_index)
        # save_frame(color_no_label, recorded_frame + "/color_no_label", frame_index)
        frame_index += 1

        cv2.imshow('Obstacle Detection (RGB)', color_image)
        cv2.imshow('Depth Map', depth_colormap)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    ## Cleanup
    # rgb_out.release()
    # depth_out.release()
    # rgb_no_label_out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_image_folder()
