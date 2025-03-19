# path_decision.py
import numpy as np
import cv2

import config

def make_decision(detected_objects, safe_distance, angle_stabilization, prev_turn_angle, img_width,FOV):
    
    command_text = "FORWARD"
    turn_angle = prev_turn_angle  # Initialize turn_angle with prev_turn_angle

    for obj in detected_objects : 
        _, _, _, _, _, _, _, _, cls = obj
        class_name = config.CLASS_NAMES[int(cls)]
        if class_name == "target" : 
            return "Target Found", turn_angle
        
    if is_forward_path_clear(detected_objects, img_width):
        return "FORWARD", turn_angle

    if detected_objects:
        closest_x, _, closest_depth, x1, y1, x2, y2, conf, cls = detected_objects[0]

        if closest_depth < safe_distance:
            turn_angle = int(((x2 - x1) / img_width) * FOV)

            if prev_turn_angle is not None and abs(turn_angle - prev_turn_angle) < angle_stabilization:
                turn_angle = prev_turn_angle

            if closest_x < img_width * 0.5:
                command_text = f"TURN RIGHT {turn_angle} degrees"
            else:
                command_text = f"TURN LEFT {turn_angle} degrees"

    if len(detected_objects) == 0 :
        return 'No Obstacle', turn_angle
    else : 
        pass

    return command_text, turn_angle

def make_depth_based_decision(depth_image, max_distance_mm, safe_distance, width, fov):

    if depth_image is None:
        return "No Depth Data"

    # Remove zero depth values (invalid readings)
    depth_values = depth_image[depth_image > 0]
    
    if depth_values.size == 0:
        return "STOP NO DEPTH DATA"

    # Compute median depth for the whole frame
    overall_depth = np.median(depth_values)

    # If there's enough space in front, continue moving forward
    if overall_depth > safe_distance:
        return "FORWARD"

    # Split depth map into left and right halves
    half_width = width // 2
    left_region = depth_image[:, :half_width]
    right_region = depth_image[:, half_width:]

    # Compute median depth in left and right regions
    left_depth = np.median(left_region[left_region > 0]) if np.any(left_region > 0) else None
    right_depth = np.median(right_region[right_region > 0]) if np.any(right_region > 0) else None


    # Move towards the side with more depth (more open space)
    if left_depth > safe_distance and right_depth > safe_distance :
        return "FORWARD"
    elif left_depth > right_depth:
        return "LEFT"
    elif right_depth > left_depth:
        return "RIGHT"

    return "STOP NO WAY OUT"

def is_forward_path_clear(detected_objects, img_width):
    """
    Check if the center path is blocked by any detected objects.
    
    Parameters:
    - detected_objects (list): List of detected objects with bounding box details.
    - img_width (int): Width of the frame.

    Returns:
    - bool: True if the forward path is clear, False if blocked.
    """
    center_start = int(img_width * 0.4)  # Define middle region (40%-60% of width)
    center_end = int(img_width * 0.6)

    for obj in detected_objects:
        _, _, _, x1, _, x2, _, _, _ = obj  # Extract bounding box coordinates

        # Check if any object is blocking the center region
        if x1 < center_end and x2 > center_start:
            return False  # Center is blocked

    return True  # No object in center region



