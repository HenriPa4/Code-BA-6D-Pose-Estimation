# --- POSE ESTIMATION VERIFICATION SCRIPT ---
# This can be a new script, or integrated after calibration

import pyrealsense2 as rs
import cv2
import cv2.aruco as aruco
import numpy as np

# --- CHARUCO BOARD SETUP (use the same as in calibration) ---
SQUARES_VERTICALLY = 3
SQUARES_HORIZONTALLY = 4
square_size = 0.18
marker_size = 0.135
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
board = aruco.CharucoBoard(
    (SQUARES_HORIZONTALLY, SQUARES_VERTICALLY), 
    square_size, 
    marker_size, 
    aruco_dict
)
charuco_detector = aruco.CharucoDetector(board)

# Load the calibration data you saved
calibration_file = "calibration_d405.npz"
with np.load(calibration_file) as X:
    camera_matrix, dist_coeffs = [X[i] for i in ('camera_matrix', 'dist_coeffs')]

# --- RealSense Setup ---
pipeline = rs.pipeline()
config = rs.config()
config.enable_device("336522302876")
config.enable_stream(rs.stream.color, 1280, 800, rs.format.bgr8, 30)
pipeline.start(config)

align = rs.align(rs.stream.color)

print("Starting pose estimation check. Move the board. ESC to quit.")
window_title = "Pose Estimation Check"
cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)
        color_frame = aligned.get_color_frame()

        if not color_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        disp = frame.copy()

        charuco_corners, charuco_ids, _, _ = charuco_detector.detectBoard(gray)

        # If the board is detected
        if charuco_ids is not None and len(charuco_ids) > 4:
            # Estimate the pose of the board
            obj_points, img_points = board.matchImagePoints(charuco_corners, charuco_ids)
            
            # The function now returns retval, rvec, tvec
            retval, rvec, tvec = cv2.solvePnP(obj_points, img_points, camera_matrix, dist_coeffs)

            if retval:
                # Project 3D points (the axes) to image plane
                cv2.drawFrameAxes(disp, camera_matrix, dist_coeffs, rvec, tvec, 0.1) # Draw axis 10cm long

        cv2.imshow(window_title, disp)
        key = cv2.waitKey(1) & 0xFF
        if key == 27: # ESC key
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()