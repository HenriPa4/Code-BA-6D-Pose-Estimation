import os, copy
import cv2
import numpy as np
import onnxruntime as rt
import threading
import queue
import time
import json
from scipy.spatial.transform import Rotation as R

from object_pose_utils_onnx import get_cuboid_corner, get_camera_matrix, get_class_names, draw_obj_pose, draw_bbox_2d

# === Config ===
MODEL_PATH = 'Models/yolox.onnx' # <-- UPDATE THIS PATH
INPUT_WIDTH, INPUT_HEIGHT = 1280, 800
SCORE_THRESH = 0.3
DATASET = 'custom'

# --- IMPORTANT: List your 3 camera indices and identify the master camera ---
CAMERA_INDICES = [4, 10, 16]      # <-- UPDATE WITH YOUR 3 CAMERA INDICES
MASTER_CAMERA_INDEX = 4          # <-- UPDATE WITH YOUR CHOSEN MASTER CAMERA INDEX

# --- Paths to your INTRINSIC calibration JSON files for each camera ---
INTRINSICS_PATHS = {
    4: 'intrinsic_calibration/camera_1.json',  # <-- UPDATE THIS
    10: 'intrinsic_calibration/camera_2.json', # <-- UPDATE THIS
    16: 'intrinsic_calibration/camera_3.json'  # <-- UPDATE THIS
}

# --- Paths to your extrinsic calibration JSON files ---
EXTRINSICS_PATHS = {
    16: 'extrinsic_calibration/images_extrinsic_calibration/old336222300607_and_337122300608/extrinsic_charuco.json', # <-- UPDATE THIS PATH
    10: 'extrinsic_calibration/images_extrinsic_calibration/old336222300607_and_336522303249/extrinsic_charuco.json', # <-- UPDATE THIS PATH
}

# === Load model ===
session = rt.InferenceSession(MODEL_PATH, providers=[
    ('CUDAExecutionProvider', {
        'device_id': 1,
        'arena_extend_strategy': 'kNextPowerOfTwo',
        'gpu_mem_limit': 2 * 1024 * 1024 * 1024,  # 2GB
        'cudnn_conv_algo_search': 'DEFAULT',
        'do_copy_in_default_stream': True,
    }),
    'CPUExecutionProvider',
])

# === Load pose metadata ===
class_to_cuboid = get_cuboid_corner(dataset=DATASET)
class_names = get_class_names(dataset=DATASET)

# === HELPER FUNCTIONS FOR ROTATION ===
def convert_6d_to_rvec(d6_rot):
    R_col1 = d6_rot[0:3]
    R_col2 = d6_rot[3:6]
    b1 = R_col1 / np.linalg.norm(R_col1)
    dot_product = np.dot(b1, R_col2)
    b2_orthogonal = R_col2 - dot_product * b1
    b2 = b2_orthogonal / np.linalg.norm(b2_orthogonal)
    b3 = np.cross(b1, b2)
    R_mat = np.stack((b1, b2, b3), axis=-1)
    rvec, _ = cv2.Rodrigues(R_mat)
    return rvec

def convert_rvec_to_6d(rvec):
    R_mat, _ = cv2.Rodrigues(rvec)
    return R_mat[:, :2].T.flatten()

# === Load Extrinsic Calibration Data ===
def load_and_prepare_extrinsics(cam_indices, master_cam_idx, paths):
    cam_to_world = {}
    world_to_cam = {}
    identity_R, identity_T = np.eye(3), np.zeros((3, 1))
    cam_to_world[master_cam_idx] = {"R": identity_R, "T": identity_T}
    world_to_cam[master_cam_idx] = {"R": identity_R, "T": identity_T}
    for idx in cam_indices:
        if idx == master_cam_idx: continue
        try:
            with open(paths[idx], 'r') as f:
                data = json.load(f)
                R_mat = np.array(data['rotation_matrix'])
                T_vec = np.array(data['translation_vector']).reshape(3, 1)
                R_inv = R_mat.T
                T_inv = -R_inv @ T_vec
                cam_to_world[idx] = {"R": R_inv, "T": T_inv}
                world_to_cam[idx] = {"R": R_mat, "T": T_vec}
            print(f"Loaded and inverted extrinsics for camera {idx}")
        except FileNotFoundError:
            print(f"FATAL ERROR: Extrinsic file not found for camera {idx} at {paths[idx]}")
            return None, None
    return cam_to_world, world_to_cam

# +++ MODIFIED: Function to load individual intrinsic parameters and build the matrix +++
def load_intrinsics(paths):
    """Loads intrinsic camera parameters from JSON files and constructs the camera matrix."""
    intrinsic_matrices = {}
    for cam_idx, path in paths.items():
        try:
            with open(path, 'r') as f:
                data = json.load(f)
                # Read individual parameters
                fx = data['fx']
                fy = data['fy']
                cx = data['cx']
                cy = data['cy']
                
                # Assemble the 3x3 camera matrix
                cam_matrix = np.array([[fx, 0, cx],
                                       [0, fy, cy],
                                       [0, 0, 1]])
                
                intrinsic_matrices[cam_idx] = cam_matrix
                print(f"Loaded and built intrinsics for camera {cam_idx}")

        except FileNotFoundError:
            print(f"FATAL ERROR: Intrinsic file not found for camera {cam_idx} at {path}")
            return None
        except KeyError as e:
            # This will catch if 'fx', 'fy', etc. are missing
            print(f"FATAL ERROR: Missing key {e} in intrinsic file {path}")
            return None
    return intrinsic_matrices


ext_cam_to_world, ext_world_to_cam = load_and_prepare_extrinsics(CAMERA_INDICES, MASTER_CAMERA_INDEX, EXTRINSICS_PATHS)
if ext_cam_to_world is None: exit()

intrinsic_matrices = load_intrinsics(INTRINSICS_PATHS)
if intrinsic_matrices is None: exit()


# --- Threading Setup ---
class CameraThread(threading.Thread):
    def __init__(self, camera_index, camera_matrix, output_queue, stop_event):
        threading.Thread.__init__(self)
        self.camera_index = camera_index
        self.camera_matrix_base = camera_matrix
        self.output_queue = output_queue
        self.stop_event = stop_event
        self.daemon = True

    def run(self):
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened(): print(f"Error: Could not open video feed for camera {self.camera_index}."); return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, INPUT_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, INPUT_HEIGHT)
        
        w, h = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        scale_x, scale_y = INPUT_WIDTH / w if w > 0 else 0, INPUT_HEIGHT / h if h > 0 else 0
        
        scaled_camera_matrix = self.camera_matrix_base.copy()
        if scale_x > 0 and scale_y > 0:
            scaled_camera_matrix[0, 0] *= scale_x
            scaled_camera_matrix[1, 1] *= scale_y
            scaled_camera_matrix[0, 2] *= scale_x
            scaled_camera_matrix[1, 2] *= scale_y
        
        while not self.stop_event.is_set():
            ret, frame = cap.read()
            if not ret: time.sleep(0.1); continue
            origin_img = cv2.resize(frame, (INPUT_WIDTH, INPUT_HEIGHT), interpolation=cv2.INTER_LINEAR)
            img = np.ascontiguousarray(origin_img.copy().transpose((2, 0, 1)), dtype=np.float32)
            outputs = session.run(None, {session.get_inputs()[0].name: img[None, :, :, :]})
            if not self.output_queue.full():
                self.output_queue.put((self.camera_index, origin_img, outputs[0], scaled_camera_matrix))
        cap.release()

# === Detection Fusion Logic ===
def fuse_detections(world_detections, distance_thresh=0.5):
    fused_detections_dict = []
    detections_by_class = {}
    for det in world_detections:
        detections_by_class.setdefault(det['class_id'], []).append(det)
    for class_id, dets in detections_by_class.items():
        if not dets: continue
        clusters = []
        for det in dets:
            joined = False
            for cluster in clusters:
                if np.linalg.norm(det['tvec_world'] - cluster[0]['tvec_world']) < distance_thresh:
                    cluster.append(det); joined = True; break
            if not joined: clusters.append([det])
        for cluster in clusters:
            if not cluster: continue
            avg_tvec = np.mean([d['tvec_world'] for d in cluster], axis=0)
            avg_rot = R.from_rotvec([d['rvec_world'].flatten() for d in cluster]).mean().as_rotvec().reshape(3, 1)
            avg_score = np.mean([d['score'] for d in cluster])
            fused_detections_dict.append({'score': avg_score, 'class_id': class_id, 'rvec': avg_rot, 'tvec': avg_tvec})
    return fused_detections_dict

# --- Main Execution ---
if __name__ == "__main__":
    num_cameras, output_queue = len(CAMERA_INDICES), queue.Queue(maxsize=len(CAMERA_INDICES) * 2)
    stop_event, threads = threading.Event(), []
    latest_results, placeholder_frame = {}, np.zeros((INPUT_HEIGHT, INPUT_WIDTH, 3), dtype=np.uint8)
    cv2.putText(placeholder_frame, 'Waiting...', (50, INPUT_HEIGHT // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    
    for cam_index in CAMERA_INDICES:
        cam_matrix = intrinsic_matrices[cam_index]
        thread = CameraThread(cam_index, cam_matrix, output_queue, stop_event)
        threads.append(thread); thread.start()
        latest_results[cam_index] = (placeholder_frame, None, None)

    view_mode, active_view_index = 0, 0
    view_mode_names = ["Grid View (Raw Detections)", "Single View (Raw)", "Fused View (Averaged)"]

    cv2.namedWindow('Multi-Camera View', cv2.WINDOW_NORMAL)

    while True:
        while not output_queue.empty():
            try:
                cam_idx, origin_img, dets, matrix = output_queue.get_nowait()
                latest_results[cam_idx] = (origin_img, dets, matrix)
            except queue.Empty:
                continue # Ignore if the queue becomes empty mid-loop

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord('w'): view_mode = (view_mode + 1) % 3
        elif key == ord('s'): view_mode = (view_mode - 1 + 3) % 3
        if view_mode == 1:
            if key == ord('d'): active_view_index = (active_view_index + 1) % num_cameras
            elif key == ord('a'): active_view_index = (active_view_index - 1 + num_cameras) % num_cameras
        
        world_detections = []
        for cam_idx, (_, dets, _) in latest_results.items():
            if dets is None: continue
            transform = ext_cam_to_world[cam_idx]
            for det in dets:
                if det[4] < SCORE_THRESH: continue
                rvec_cam = convert_6d_to_rvec(det[6:12]).reshape(3,1)
                tvec_cam = det[12:15].reshape(3, 1)
                R_cam, _ = cv2.Rodrigues(rvec_cam)
                rvec_world, _ = cv2.Rodrigues(transform['R'] @ R_cam)
                tvec_world = transform['R'] @ tvec_cam + transform['T']
                world_detections.append({'class_id': int(det[5]), 'score': det[4], 'rvec_world': rvec_world, 'tvec_world': tvec_world})
        
        fused_world_dets = fuse_detections(world_detections)
        
        final_view = None
        if view_mode == 0 or view_mode == 2:
            processed_frames = {}
            for cam_idx in CAMERA_INDICES:
                frame, dets, matrix = latest_results[cam_idx]
                display_img = frame.copy()
                if view_mode == 0 and dets is not None and dets.shape[0] > 0:
                    dets_to_draw = dets[dets[:, 4] > SCORE_THRESH]
                    if dets_to_draw.shape[0] > 0:
                        dets_copy = dets_to_draw.copy()
                        dets_copy[:, 12:15] *= 100.0
                        draw_obj_pose(display_img, dets_copy, class_names=class_names, class_to_cuboid=class_to_cuboid, camera_matrix=matrix)
                elif view_mode == 2 and fused_world_dets:
                    draw_array = np.zeros((len(fused_world_dets), 15))
                    transform = ext_world_to_cam[cam_idx]
                    for i, det_dict in enumerate(fused_world_dets):
                        R_world, _ = cv2.Rodrigues(det_dict['rvec'])
                        rvec_cam, _ = cv2.Rodrigues(transform['R'] @ R_world)
                        tvec_cam = transform['R'] @ det_dict['tvec'] + transform['T']
                        draw_array[i, 4] = det_dict['score']
                        draw_array[i, 5] = det_dict['class_id']
                        draw_array[i, 6:12] = convert_rvec_to_6d(rvec_cam)
                        draw_array[i, 12:15] = tvec_cam.flatten()
                        draw_array[i, 12:15] *= 100.0
                    if draw_array.shape[0] > 0:
                        draw_obj_pose(display_img, draw_array, class_names=class_names, class_to_cuboid=class_to_cuboid, camera_matrix=matrix)
                cv2.putText(display_img, f"Cam: {cam_idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                processed_frames[cam_idx] = display_img
            frame1 = processed_frames.get(CAMERA_INDICES[0], placeholder_frame)
            frame2 = processed_frames.get(CAMERA_INDICES[1], placeholder_frame)
            frame3 = processed_frames.get(CAMERA_INDICES[2], placeholder_frame) if len(CAMERA_INDICES) > 2 else placeholder_frame
            frame4 = placeholder_frame
            top_row = np.hstack((frame1, frame2))
            bottom_row = np.hstack((frame3, frame4))
            final_view = np.vstack((top_row, bottom_row))
            cv2.putText(final_view, f"Mode: {view_mode_names[view_mode]} ('w'/'s')", (10, final_view.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

        elif view_mode == 1:
            display_cam_idx = CAMERA_INDICES[active_view_index]
            frame, dets, matrix = latest_results[display_cam_idx]
            img_with_pose, img_with_2d_bbox = frame.copy(), frame.copy()
            if dets is not None and dets.shape[0] > 0 and matrix is not None:
                dets_to_draw = dets[dets[:, 4] > SCORE_THRESH]
                if dets_to_draw.shape[0] > 0:
                    draw_bbox_2d(img_with_2d_bbox, dets_to_draw, class_names)
                    dets_copy = dets_to_draw.copy()
                    dets_copy[:, 12:15] *= 100.0
                    draw_obj_pose(img_with_pose, dets_copy, class_names=class_names, class_to_cuboid=class_to_cuboid, camera_matrix=matrix)
            final_view = np.hstack((img_with_2d_bbox, img_with_pose))
            cv2.putText(final_view, f"Mode: {view_mode_names[view_mode]} | Cam: {display_cam_idx} ('a'/'d')", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        
        if final_view is not None: cv2.imshow('Multi-Camera View', final_view)
    
    print("Shutting down...")
    stop_event.set()
    for thread in threads: thread.join(timeout=1.0)
    cv2.destroyAllWindows()