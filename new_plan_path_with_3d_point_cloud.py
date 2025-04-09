import numpy as np
#from get_3d_coordinates_dev import get_3D_coordinates_and_rgb

'''ค้นหาเส้นทางเมื่อมีสิ่งกีดขวางในระยะน้อยกว่า SAFE_DISTANCE '''

def detect_obstacles_and_plan_path(X, Y, Z, fov_x, fov_y, x_bounds, y_bounds, SAFE_DISTANCE, ANGLE_STABILIZATION, prev_turn_angle):
    fov_x  = int(fov_x)
    fov_y  = int(fov_y)
    x_min, x_max = x_bounds
    y_min, y_max = y_bounds

    obstacle_mask = Z < SAFE_DISTANCE
    within_x_bounds = (X >= x_min) & (X <= x_max)
    within_y_bounds = (Y >= y_min) & (Y <= y_max)
    obstacle_in_path = np.any(obstacle_mask & within_x_bounds & within_y_bounds)
    command_text = "FORWARD"
    new_turn_angle = 0

    if obstacle_in_path or prev_turn_angle > 1:
        angle_step = 1  # degrees
        angles = np.arange(angle_step, fov_x / 2, angle_step)
        
        path_found = False

        for angle in angles:
            rad = np.deg2rad(angle)
            #print("rad: ",rad)
            # Check left
            shift_x_left = X + Z * np.tan(rad)
            left_clear = not np.any((Z < (1.25)*SAFE_DISTANCE) &
                                    (shift_x_left >= x_min) & (shift_x_left <= x_max) &
                                    within_y_bounds)
            if left_clear:
                new_turn_angle = -angle
                command_text = f"TURN LEFT {angle} degrees"
                path_found = True
                break

            # Check right
            shift_x_right = X - Z * np.tan(rad)
            right_clear = not np.any((Z < (1.25)*SAFE_DISTANCE) &
                                     (shift_x_right >= x_min) & (shift_x_right <= x_max) &
                                     within_y_bounds)
            if right_clear:
                new_turn_angle = angle
                command_text = f"TURN RIGHT {angle} degrees"
                path_found = True
                break

        if not path_found:
            new_turn_angle = int(fov_x/2 )
            command_text = f"No clear path. TURN RIGHT/LEFT {int(fov_x/2)} degrees"

        # Angle stabilization (ปรับแต่งตรงนี้)
        if prev_turn_angle is not None:
            angle_difference = new_turn_angle - prev_turn_angle
            if abs(angle_difference) > ANGLE_STABILIZATION:
                turn_angle = new_turn_angle
            else:
                turn_angle = prev_turn_angle + np.sign(angle_difference) * min(abs(angle_difference), ANGLE_STABILIZATION)
        else:
            turn_angle = new_turn_angle
    else:
        turn_angle = 0
        command_text = "FORWARD"
    print(command_text)
    return command_text, turn_angle

# # Example usage after calling your existing function
# X, Y, Z,fov_x,fov_y = get_3D_coordinates_and_rgb()  # existing function

# # Cross-sectional bounds (in meters)
# x_bounds = (-0.15, 0.15)
# y_bounds = (-0.15, 0.05)

# # Safe distance threshold (in meters)
# SAFE_DISTANCE = 2.0
# prev_turn_angle = None  # Initialize the previous turn angle
# ANGLE_STABILIZATION = 5  # Prevent angle fluctuation by allowing minor variations

# # Run obstacle detection and avoidance planning
# command_text, turn_angle = detect_obstacles_and_plan_path(X, Y, Z, fov_x, fov_y, x_bounds, y_bounds, SAFE_DISTANCE, ANGLE_STABILIZATION, prev_turn_angle)
# print(command_text)
# print(turn_angle)
