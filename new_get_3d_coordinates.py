import pyrealsense2 as rs
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def get_3D_coordinates_and_rgb():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = pipeline.start(config)

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    depth_intrinsics = rs.video_stream_profile(profile.get_stream(rs.stream.depth)).get_intrinsics()
    fov_x = 2 * np.degrees(np.arctan(depth_intrinsics.width / (2 * depth_intrinsics.fx)))
    fov_y = 2 * np.degrees(np.arctan(depth_intrinsics.height / (2 * depth_intrinsics.fy)))

    try:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            raise RuntimeError("No frames received")

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        mask = (depth_image == 0).astype(np.uint8) * 255
        inpainted_depth = cv.inpaint(depth_image, mask, 3, cv.INPAINT_NS)
        depth_display = cv.normalize(inpainted_depth, None, 0, 255, cv.NORM_MINMAX)
        depth_display = np.uint8(depth_display)

        x = np.linspace(0, depth_intrinsics.width - 1, depth_intrinsics.width)
        y = np.linspace(0, depth_intrinsics.height - 1, depth_intrinsics.height)
        grid_x, grid_y = np.meshgrid(x, y)
        depth = inpainted_depth[grid_y.astype(np.int32), grid_x.astype(np.int32)] * depth_scale
        X = (grid_x - depth_intrinsics.ppx) / depth_intrinsics.fx * depth
        Y = (grid_y - depth_intrinsics.ppy) / depth_intrinsics.fy * depth
        Z = depth

        valid = depth > 0
        X = X[valid]
        Y = Y[valid]
        Z = Z[valid]
        # for i in range(len(X)):
        #     if( Z[i] <= 0.3  ):
        #         print('[' , X[i] , ',' , Y[i] , ',',Z[i], ']')
        #     else:
        #         pass

        '''plot fig for check'''
        # fig = plt.figure(figsize=(18, 6))
        
        # ax0 = fig.add_subplot(131)
        # img = ax0.imshow(depth_display, cmap='jet')
        # ax0.set_title('Inpainted Depth Map')
        # ax0.set_xlabel('Pixel X')
        # ax0.set_ylabel('Pixel Y')
        # plt.colorbar(img, ax=ax0, orientation='vertical', label='Depth scale (Color)')
        
        # ax1 = fig.add_subplot(132)
        # ax1.imshow(color_image)
        # ax1.set_title('RGB Image')
        
        # ax2 = fig.add_subplot(133, projection='3d')
        # scatter = ax2.scatter(X, Y, Z, c=-Z, cmap='hot', marker='.')
        # ax2.set_title('3D Point Cloud')
        # ax2.set_xlabel('X (m)')
        # ax2.set_ylabel('Y (m)')
        # ax2.set_zlabel('Depth (m)')
        # ax2.view_init(elev=-180, azim=90, roll=180)  # Adjust to match the front view of depth map
        # plt.show()

        return X, Y, Z,fov_x,fov_y

    finally:
        pipeline.stop()
        # print("Horizontal FoV: {:.2f} degrees".format(fov_x))
        # print("Vertical FoV: {:.2f} degrees".format(fov_y))
        # print("depth_scale: ",depth_scale)
        # print("width: ",depth_intrinsics.width)
        # print(depth_intrinsics.height)
        # print("ppx: ",depth_intrinsics.ppx)
        # print("focus x: " ,depth_intrinsics.fx)
        # print("ppy: ", depth_intrinsics.ppy)
        # print("focus y: " , depth_intrinsics.fy)

# Example usage
#X, Y, Z,fov_x,fov_y = get_3D_coordinates_and_rgb()
