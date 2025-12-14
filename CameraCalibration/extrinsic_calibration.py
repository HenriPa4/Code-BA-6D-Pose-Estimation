#!/usr/bin/env python3

import os
import cv2
import cv2.aruco as aruco
import json
import time
import numpy as np
import shutil
import pyrealsense2 as rs
import config  # our configuration file

# Map the ARUCO_DICT string from config to an actual dictionary:
ARUCO_DICT_MAP = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    # add other dictionaries as needed
}
aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_MAP[config.ARUCO_DICT])

def list_realsense_cameras():
    ctx = rs.context()
    devices = ctx.query_devices()
    camera_list = []
    for device in devices:
        serial = device.get_info(rs.camera_info.serial_number)
        rgb_stream_found = False
        for sensor in device.query_sensors():
            for profile in sensor.get_stream_profiles():
                if profile.stream_type() == rs.stream.color:
                    rgb_stream_found = True
                    break
            if rgb_stream_found:
                break
        if rgb_stream_found:
            camera_list.append({'serial': serial})
    return camera_list

def select_two_cameras_interactively(camera_list):
    if len(camera_list) < 2:
        print("[ERROR] At least two RealSense cameras are required for extrinsic calibration.")
        return None, None

    print("\n=== Available RealSense Cameras ===")
    for idx, cam in enumerate(camera_list):
        print(f"[{idx}] Serial Number: {cam['serial']}")

    serials = [None, None]
    while True:
        try:
            selection1 = int(input("\nEnter the number for the first camera: "))
            if 0 <= selection1 < len(camera_list):
                serials[0] = camera_list[selection1]['serial']
                break
            else:
                print("[WARNING] Invalid selection. Try again.")
        except ValueError:
            print("[WARNING] Please enter a valid integer.")

    while True:
        try:
            selection2 = int(input("Enter the number for the second camera: "))
            if 0 <= selection2 < len(camera_list) and selection2 != selection1:
                serials[1] = camera_list[selection2]['serial']
                break
            else:
                print("[WARNING] Invalid or duplicate selection. Try again.")
        except ValueError:
            print("[WARNING] Please enter a valid integer.")

    return serials[0], serials[1]

def configure_realsense_pipeline(serial):
    pipeline = rs.pipeline()
    config_rs = rs.config()
    config_rs.enable_device(serial)
    config_rs.enable_stream(rs.stream.color, config.CAMERA_WIDTH, config.CAMERA_HEIGHT, rs.format.bgr8, config.CAMERA_FPS)
    try:
        pipeline.start(config_rs)
    except Exception as e:
        print(f"[ERROR] Failed to start pipeline for serial {serial}: {e}")
        return None
    return pipeline

def load_intrinsic_calibration(json_path):
    if not os.path.exists(json_path):
        print(f"[ERROR] Intrinsic calibration JSON not found: {json_path}")
        return None
    with open(json_path, 'r') as f:
        data = json.load(f)
    try:
        camera_matrix = np.array(data["camera_matrix"], dtype=np.float32)
        dist_coeffs   = np.array(data["dist_coeff"], dtype=np.float32)
        w = int(data["image_width"])
        h = int(data["image_height"])
        return camera_matrix, dist_coeffs, w, h
    except KeyError as e:
        print(f"[ERROR] Missing key in JSON file: {e}")
        return None

def vec_to_rotation_matrix(rvec):
    R, _ = cv2.Rodrigues(rvec)
    return R

def compute_camera_to_camera_transform(rvec1, tvec1, rvec2, tvec2):
    R1 = cv2.Rodrigues(rvec1)[0]
    R2 = cv2.Rodrigues(rvec2)[0]
    R_12 = R2 @ R1.T
    T_12 = tvec2 - R2 @ R1.T @ tvec1
    return R_12, T_12

def main():
    # Use hyperparameters from config.py
    output_dir = config.EXTRINSIC_DIR
    burst_size = config.BURST_SIZE

    # ------------------------ ChArUco Board Definition ------------------------
    squares_x = config.SQUARES_X
    squares_y = config.SQUARES_Y
    square_length = config.SQUARE_LENGTH
    marker_length = config.MARKER_LENGTH
    board = aruco.CharucoBoard((squares_x, squares_y), square_length, marker_length, aruco_dict)
    detector_params = aruco.DetectorParameters()
    detector = aruco.CharucoDetector(board)


    # ------------------------ Discover & Pick Cameras ------------------------
    camera_list = list_realsense_cameras()
    serial1, serial2 = select_two_cameras_interactively(camera_list)
    if not serial1 or not serial2:
        print("[ERROR] Camera selection failed. Exiting.")
        return
    print(f"[INFO] Camera 1: {serial1}")
    print(f"[INFO] Camera 2: {serial2}")

    # ------------------------ Load Intrinsic Calibration ------------------------
    intr1_path = os.path.join(config.INTRINSIC_DIR, f"camera_{serial1}", "camera_calibration.json")
    intr2_path = os.path.join(config.INTRINSIC_DIR, f"camera_{serial2}", "camera_calibration.json")
    intr1 = load_intrinsic_calibration(intr1_path)
    intr2 = load_intrinsic_calibration(intr2_path)
    if intr1 is None or intr2 is None:
        print("[ERROR] Failed to load intrinsic calibration for one or both cameras.")
        return
    cam_mat1, dist1, w1, h1 = intr1
    cam_mat2, dist2, w2, h2 = intr2

    # ------------------------ Start Pipelines & Prepare Folders ------------------------
    pipeline1 = configure_realsense_pipeline(serial1)
    pipeline2 = configure_realsense_pipeline(serial2)
    if pipeline1 is None or pipeline2 is None:
        print("[ERROR] Failed to start pipeline(s). Exiting.")
        return

    pair_dir = os.path.join(output_dir, f"{serial1}_and_{serial2}")
    cam1_dir = os.path.join(pair_dir, f"cam_{serial1}")
    cam2_dir = os.path.join(pair_dir, f"cam_{serial2}")
    if os.path.exists(pair_dir):
        shutil.rmtree(pair_dir)
        print(f"[INFO] Cleared existing data in '{pair_dir}'.")
    os.makedirs(cam1_dir, exist_ok=True)
    os.makedirs(cam2_dir, exist_ok=True)
    print(f"[INFO] Created directory for image capture: {pair_dir}")

    print("\n=== Extrinsic Calibration: ChArUco Board ===")
    print("Instructions:")
    print("  - Place the ChArUco board within view of both cameras.")
    print("  - When the board is detected, a coordinate system will be overlaid on each view.")
    print("    The axes are drawn 5 times larger than default and shifted so the origin is at the board's center.")
    print(f"  - Press 'c' or 'p' to capture a burst of {burst_size} images from each camera.")
    print("  - Press 'q' to finish capturing and compute the extrinsic calibration.\n")

    # ------------------------ Live Capture with Axis Overlay & Resized Display ------------------------
    image_counter = 0
    while True:
        frames1 = pipeline1.wait_for_frames()
        frames2 = pipeline2.wait_for_frames()
        color_image1 = np.asanyarray(frames1.get_color_frame().get_data())
        color_image2 = np.asanyarray(frames2.get_color_frame().get_data())

        # Draw coordinate axes on each image (if board detected)
        for (img, cam_mat, dist) in [(color_image1, cam_mat1, dist1), (color_image2, cam_mat2, dist2)]:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            corners, ids, marker_corners, marker_ids = detector.detectBoard(gray)
            if ids is not None and len(ids) >= 4:
                objPoints, imgPoints = board.matchImagePoints(corners, ids)
                ret_pose, rvec, tvec = cv2.solvePnP(objPoints, imgPoints, cam_mat, dist, None, None)
                if ret_pose:
                    cv2.drawFrameAxes(img, cam_mat, dist, rvec, tvec, square_length * 2.5)

        # For live preview, resize images to the display resolution from config.py
        disp1 = cv2.resize(color_image1, (config.DISPLAY_WIDTH, config.DISPLAY_HEIGHT))
        disp2 = cv2.resize(color_image2, (config.DISPLAY_WIDTH, config.DISPLAY_HEIGHT))
        combined = np.hstack((disp1, disp2))
        cv2.imshow("Camera 1 (left) | Camera 2 (right)", combined)
        key = cv2.waitKey(1) & 0xFF

        if key in [ord('c'), ord('p')]:
            # Capture a burst of frames from each camera (save full-resolution images)
            for i in range(burst_size):
                f1 = pipeline1.wait_for_frames()
                f2 = pipeline2.wait_for_frames()
                ci1 = np.asanyarray(f1.get_color_frame().get_data())
                ci2 = np.asanyarray(f2.get_color_frame().get_data())
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                img1_name = f"burst_{image_counter:03d}_{i}_{timestamp}_{serial1}.png"
                img2_name = f"burst_{image_counter:03d}_{i}_{timestamp}_{serial2}.png"
                cv2.imwrite(os.path.join(cam1_dir, img1_name), ci1)
                cv2.imwrite(os.path.join(cam2_dir, img2_name), ci2)
            print(f"[INFO] Captured burst #{image_counter} of {burst_size} frames from each camera.")
            image_counter += 1
        elif key == ord('q'):
            print("[INFO] Quitting capture loop; starting extrinsic calibration.")
            break

    pipeline1.stop()
    pipeline2.stop()
    cv2.destroyAllWindows()

    # ------------------------ Post-Capture: Process Captured Images ------------------------
    images_cam1 = sorted([f for f in os.listdir(cam1_dir) if f.lower().endswith(('.png','.jpg','.jpeg'))])
    images_cam2 = sorted([f for f in os.listdir(cam2_dir) if f.lower().endswith(('.png','.jpg','.jpeg'))])

    def get_burst_id(fname):
        base = os.path.splitext(fname)[0]
        parts = base.split('_')
        if len(parts) < 3 or parts[0] != "burst":
            return None
        return parts[1]

    cam1_bursts = {}
    for f in images_cam1:
        bid = get_burst_id(f)
        if bid is not None:
            cam1_bursts.setdefault(bid, []).append(f)
    cam2_bursts = {}
    for f in images_cam2:
        bid = get_burst_id(f)
        if bid is not None:
            cam2_bursts.setdefault(bid, []).append(f)

    # For each burst common to both cameras, detect board pose and compute relative transforms.
    valid_transforms = []
    common_bursts = set(cam1_bursts.keys()).intersection(set(cam2_bursts.keys()))
    if not common_bursts:
        print("[ERROR] No matching bursts between the two cameras.")
        return

    def detect_charuco_pose(img_path, cam_mtx, dist_coeff):
        img = cv2.imread(img_path)
        if img is None:
            return None
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(gray)
        if charuco_ids is None or len(charuco_ids) < 4:
            return None
        objPoints, imgPoints = board.matchImagePoints(charuco_corners, charuco_ids)
        ret_pose, rvec, tvec = cv2.solvePnP(objPoints, imgPoints, cam_mtx, dist_coeff, None, None)
        if not ret_pose:
            return None
        return (rvec, tvec)

    for burst_id in sorted(common_bursts):
        files1 = cam1_bursts[burst_id]
        files2 = cam2_bursts[burst_id]
        pair_count = min(len(files1), len(files2))
        for i in range(pair_count):
            img1_path = os.path.join(cam1_dir, files1[i])
            img2_path = os.path.join(cam2_dir, files2[i])
            pose1 = detect_charuco_pose(img1_path, cam_mat1, dist1)
            pose2 = detect_charuco_pose(img2_path, cam_mat2, dist2)
            if pose1 is None or pose2 is None:
                continue
            rvec1, tvec1 = pose1
            rvec2, tvec2 = pose2
            R_12, T_12 = compute_camera_to_camera_transform(rvec1, tvec1, rvec2, tvec2)
            valid_transforms.append((R_12, T_12))

    if len(valid_transforms) < 1:
        print("[ERROR] No valid ChArUco detections found across bursts. Exiting.")
        return

    # ------------------------ Outlier Filtering ------------------------
    rod_vecs = []
    trans_vecs = []
    for (R12, T12) in valid_transforms:
        rvec, _ = cv2.Rodrigues(R12)
        rod_vecs.append(rvec.reshape(3))
        trans_vecs.append(T12.reshape(3))
    rod_arr = np.array(rod_vecs)
    trans_arr = np.array(trans_vecs)
    median_rod = np.median(rod_arr, axis=0)
    median_trans = np.median(trans_arr, axis=0)
    threshold_rot = 0.4    # radians (~23°)
    threshold_trans = 0.05  # meters (5 cm)
    keep_indices = []
    for i in range(len(rod_arr)):
        err_rot = np.linalg.norm(rod_arr[i] - median_rod)
        err_trans = np.linalg.norm(trans_arr[i] - median_trans)
        if err_rot < threshold_rot and err_trans < threshold_trans:
            keep_indices.append(i)
    if len(keep_indices) == 0:
        print("[ERROR] All detections were filtered out as outliers. Exiting.")
        return
    filtered_rod = rod_arr[keep_indices]
    filtered_trans = trans_arr[keep_indices]
    print(f"[INFO] Filtering: kept {len(keep_indices)} out of {len(rod_arr)} detections.")

    # ------------------------ Average Transforms & Compute RMS Error ------------------------
    mean_rod = np.mean(filtered_rod, axis=0)
    mean_trans = np.mean(filtered_trans, axis=0)
    mean_R, _ = cv2.Rodrigues(mean_rod)
    rms_rot = np.sqrt(np.mean(np.square(np.linalg.norm(filtered_rod - mean_rod, axis=1))))
    rms_trans = np.sqrt(np.mean(np.square(np.linalg.norm(filtered_trans - mean_trans, axis=1))))
    rms_rot_deg = np.degrees(rms_rot)

    print("\n=== Extrinsic Calibration Results ===")
    print(f"Number of valid detections (after filtering): {len(filtered_rod)}")
    print("Final Rotation (Camera1 -> Camera2):\n", mean_R)
    print("Final Translation (Camera1 -> Camera2):\n", mean_trans)
    print(f"RMS Rotation Error: {rms_rot_deg:.2f} degrees")
    print(f"RMS Translation Error: {rms_trans:.4f} meters")

    extrinsic_json = {
        "camera_1_serial": serial1,
        "camera_2_serial": serial2,
        "num_detections": len(filtered_rod),
        "rotation_matrix": mean_R.tolist(),
        "translation_vector": mean_trans.tolist(),
        "rms_rotation_error_degrees": rms_rot_deg,
        "rms_translation_error_m": float(rms_trans)
    }
    output_json_path = os.path.join(pair_dir, "extrinsic_charuco.json")
    with open(output_json_path, "w") as f:
        json.dump(extrinsic_json, f, indent=4)
    print(f"\n[INFO] Extrinsic calibration saved to: {output_json_path}")
    print("[INFO] Calibration completed successfully!\n")

if __name__ == "__main__":
    main()

