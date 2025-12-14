import os
import cv2
import cv2.aruco as aruco
import numpy as np
import json
import glob
import onnxruntime as rt
from scipy.spatial.transform import Rotation as R
from object_pose_utils_onnx import draw_cuboid_2d

# ================= CONFIGURATION =================
IMAGE_FOLDER = "Pictures"
OUTPUT_FOLDER = "Evaluation_Fusion_Results"
MODEL_PATH = 'Models/yolox.onnx' 
INPUT_WIDTH, INPUT_HEIGHT = 1280, 800
CONF_THRES = 0.3

# --- Camera Mapping ---
PREFIX_TO_INDEX = {'1': 4, '2': 10, '3': 16}
CAMERA_INDICES = [4, 10, 16] # Order for display
MASTER_CAMERA_INDEX = 4

# --- Paths ---
INTRINSICS_PATHS = {
    4: 'camera_1.json',
    10: 'camera_2.json',
    16: 'camera_3.json'
}

EXTRINSICS_PATHS = {
    16: 'extrinsic_charuco336522303249.json',
    10: 'extrinsic_charuco337122300608.json',
}

# --- Bias Correction (From your analysis) ---
CAMERA_BIAS = {
    4:  np.array([-0.0099, 0.0, 0.0]).reshape(3, 1),
    10: np.array([-0.0164, 0.0, 0.0]).reshape(3, 1),
    16: np.array([-0.0022, 0.0, 0.0]).reshape(3, 1)
}

# --- Physical Definitions (Meters) ---
ARUCO_MARKER_SIZE_METERS = 0.18
CUBE_HALF_EXTENT = 0.1
OBJ_DIM_X, OBJ_DIM_Y, OBJ_DIM_Z = 0.71, 0.69, 1.02
ID_TOP = 10

# =================================================

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
stats_file_path = os.path.join(OUTPUT_FOLDER, "fusion_metrics.json")
session = rt.InferenceSession(MODEL_PATH, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

# --- 1. Helper Functions ---

def create_ref_transform(rot_axis, angle_deg, translation):
    r = R.from_euler(rot_axis, angle_deg, degrees=True)
    mat = np.eye(4)
    mat[:3, :3] = r.as_matrix()
    mat[:3, 3] = translation
    return mat

# Aruco Definitions
T_REF_FRONT = create_ref_transform('y', 0,   [0, 0, CUBE_HALF_EXTENT])
T_REF_RIGHT = create_ref_transform('y', 90,  [CUBE_HALF_EXTENT, 0, 0])
T_REF_BACK  = create_ref_transform('y', 180, [0, 0, -CUBE_HALF_EXTENT])
T_REF_LEFT  = create_ref_transform('y', -90, [-CUBE_HALF_EXTENT, 0, 0])
T_REF_TOP   = create_ref_transform('x', -90, [0, -CUBE_HALF_EXTENT, 0])
MARKER_POSES = {11: T_REF_FRONT, 12: T_REF_RIGHT, 13: T_REF_BACK, 14: T_REF_LEFT, 10: T_REF_TOP}

# Object Offset
rot_matrix = create_ref_transform('zx', [180, -90], [0,0,0])
x_offset = CUBE_HALF_EXTENT + 0.39 - (OBJ_DIM_X / 2.0)
y_offset = CUBE_HALF_EXTENT + 0.27 - (OBJ_DIM_Y / 2.0)
z_offset = - ((OBJ_DIM_Z / 2.0) + 0.05)
trans_matrix = np.eye(4); trans_matrix[:3, 3] = [x_offset, y_offset, z_offset]
T_REF_OBJECT = rot_matrix @ trans_matrix

def get_cuboid_verts_meters():
    w, h, d = OBJ_DIM_X/2, OBJ_DIM_Y/2, OBJ_DIM_Z/2
    return np.array([[-w,-h,-d], [-w,-h,d], [-w,h,d], [-w,h,-d], [w,-h,-d], [w,-h,d], [w,h,d], [w,h,-d]], dtype=np.float32)
CUBIOD_VERTS = get_cuboid_verts_meters()

def convert_6d_to_rvec(d6_rot):
    a1, a2 = d6_rot[:3], d6_rot[3:]
    b1 = a1 / np.linalg.norm(a1)
    b2 = a2 - np.dot(b1, a2) * b1; b2 /= np.linalg.norm(b2)
    b3 = np.cross(b1, b2)
    return cv2.Rodrigues(np.stack((b1, b2, b3), axis=1))[0]

# --- 2. Calibration Loading ---
def load_calibration():
    intrinsics = {}
    extrinsics_cam_to_world = {} # This is T_world_to_cam (World -> Cam)
    
    for idx, path in INTRINSICS_PATHS.items():
        with open(path, 'r') as f: d = json.load(f)
        intrinsics[idx] = np.array([[d['fx'], 0, d['cx']], [0, d['fy'], d['cy']], [0, 0, 1]])
        
    identity = np.eye(4)
    extrinsics_cam_to_world[MASTER_CAMERA_INDEX] = identity 
    
    for idx, path in EXTRINSICS_PATHS.items():
        with open(path, 'r') as f: d = json.load(f)
        R_mat = np.array(d['rotation_matrix'])
        T_vec = np.array(d['translation_vector']).reshape(3, 1)
        # We need the matrix that transforms World Point -> Cam Point
        # The JSON usually gives Cam -> World or World -> Cam.
        # Based on your live feed script logic:
        # cam_to_world[idx] = {"R": R_inv, "T": T_inv} 
        # This implies the JSON is Cam -> World.
        # So World -> Cam (which we need for projection) is the raw matrix in JSON.
        
        # NOTE: Your LiveFeed script does: transform['R'] @ tvec_cam + transform['T']
        # This implies transform is Cam -> World.
        # So we need the inverse of that to project World -> Cam.
        
        R_inv = R_mat.T
        T_inv = -R_inv @ T_vec
        
        # This matrix converts Cam -> World
        cam_to_world = np.eye(4)
        cam_to_world[:3, :3] = R_inv
        cam_to_world[:3, 3] = T_inv.flatten()
        
        # For projection, we need World -> Cam (Inverse of Cam->World)
        extrinsics_cam_to_world[idx] = np.linalg.inv(cam_to_world)
        
    # Special handling for master (Identity)
    # Master is World. So World->Master is Identity.
    extrinsics_cam_to_world[MASTER_CAMERA_INDEX] = np.eye(4)
        
    return intrinsics, extrinsics_cam_to_world, {idx: np.zeros(5) for idx in intrinsics}

# --- 3. Processing Functions ---

def get_aruco_pose_world(img, cam_idx, intrinsics, extrinsics, dist_coeffs):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    try: params = aruco.DetectorParameters()
    except: params = aruco.DetectorParameters_create()
    params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    
    try: detector = aruco.ArucoDetector(aruco_dict, params); corners, ids, _ = detector.detectMarkers(gray)
    except: corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=params)
    
    if ids is None: return None

    marker_points = np.array([[-0.09, 0.09, 0], [0.09, 0.09, 0], [0.09, -0.09, 0], [-0.09, -0.09, 0]], dtype=np.float32)
    candidates = []
    
    for i, marker_id in enumerate(ids.flatten()):
        if marker_id not in MARKER_POSES: continue
        _, rvec, tvec = cv2.solvePnP(marker_points, corners[i], intrinsics[cam_idx], dist_coeffs[cam_idx], False, cv2.SOLVEPNP_IPPE_SQUARE)
        
        R_mat, _ = cv2.Rodrigues(rvec)
        T_cam_marker = np.eye(4); T_cam_marker[:3, :3] = R_mat; T_cam_marker[:3, 3] = tvec.flatten()
        T_cam_ref = T_cam_marker @ np.linalg.inv(MARKER_POSES[marker_id])
        candidates.append({"id": marker_id, "T": T_cam_ref})

    if not candidates: return None

    candidates.sort(key=lambda x: 0 if x["id"] == ID_TOP else 1)
    T_base = candidates[0]["T"]
    valid_Ts = [T_base]
    for k in range(1, len(candidates)):
        R_diff = T_base[:3,:3].T @ candidates[k]["T"][:3,:3]
        if np.trace(R_diff) < 0.5: valid_Ts.append(T_base)
        else: valid_Ts.append(candidates[k]["T"])

    avg_t = np.mean([t[:3, 3] for t in valid_Ts], axis=0)
    qs = [R.from_matrix(t[:3, :3]).as_quat() for t in valid_Ts]
    base_q = qs[0]
    for k in range(1, len(qs)): 
        if np.dot(base_q, qs[k]) < 0: qs[k] *= -1
    avg_R = R.from_quat(np.mean(qs, axis=0) / np.linalg.norm(np.mean(qs, axis=0))).as_matrix()
    
    T_cam_ref_avg = np.eye(4); T_cam_ref_avg[:3, :3] = avg_R; T_cam_ref_avg[:3, 3] = avg_t
    T_cam_obj = T_cam_ref_avg @ T_REF_OBJECT
    
    # We stored World -> Cam in extrinsics. So Cam -> World is Inverse.
    T_cam_to_world = np.linalg.inv(extrinsics[cam_idx])
    T_world_obj = T_cam_to_world @ T_cam_obj
    return T_world_obj

def get_model_prediction_world(img, cam_idx, intrinsics, extrinsics):
    input_img = cv2.resize(img, (INPUT_WIDTH, INPUT_HEIGHT))
    blob = np.ascontiguousarray(input_img.transpose((2, 0, 1))[None, :, :, :], dtype=np.float32)
    
    outputs = session.run(None, {session.get_inputs()[0].name: blob})
    dets = outputs[0]
    if dets.ndim == 3: dets = dets[0]
    if dets.ndim == 1: dets = dets[None, :]
    
    best_det = None
    best_score = -1
    for det in dets:
        if det[4] > CONF_THRES and det[4] > best_score:
            best_score = det[4]
            best_det = det
            
    if best_det is None: return None

    rvec_cam = convert_6d_to_rvec(best_det[6:12])
    tvec_cam = best_det[12:15].reshape(3, 1)

    # Correction
    bias = CAMERA_BIAS.get(cam_idx, np.zeros((3,1)))
    tvec_cam = tvec_cam - bias

    # Transform to World
    T_cam_to_world = np.linalg.inv(extrinsics[cam_idx])
    
    R_cam, _ = cv2.Rodrigues(rvec_cam)
    T_cam_obj = np.eye(4)
    T_cam_obj[:3, :3] = R_cam
    T_cam_obj[:3, 3] = tvec_cam.flatten()
    
    T_world_obj = T_cam_to_world @ T_cam_obj
    
    # Extract World components
    rvec_world, _ = cv2.Rodrigues(T_world_obj[:3, :3])
    tvec_world = T_world_obj[:3, 3]
    
    return {
        'class_id': 0, 'score': best_score,
        'rvec_world': rvec_world,
        'tvec_world': tvec_world.flatten()
    }

def calculate_error(T_gt, T_pred):
    dist = np.linalg.norm(T_gt[:3, 3] - T_pred[:3, 3])
    R_diff = T_gt[:3, :3].T @ T_pred[:3, :3]
    trace = max(-1, min(3, np.trace(R_diff)))
    angle = np.degrees(np.arccos((trace - 1) / 2))
    return dist, angle

# --- 4. Main Loop ---

def main():
    # Load World->Cam matrices
    intrinsic_mats, world_to_cam_mats, dist_coeffs = load_calibration()
    
    files = sorted(glob.glob(os.path.join(IMAGE_FOLDER, "*.jpg")) + glob.glob(os.path.join(IMAGE_FOLDER, "*.png")))
    groups = {} 
    
    for f in files:
        base = os.path.basename(f)
        parts = base.split('_')
        if len(parts) < 2: continue
        prefix, img_id = parts[0], parts[1]
        if img_id not in groups: groups[img_id] = {}
        groups[img_id][prefix] = f
        
    results = []
    print(f"Found {len(groups)} potential image sets.")

    for img_id, cam_files in groups.items():
        if len(cam_files) < 2: continue
        
        gt_poses_world = []
        pred_dicts_world = []
        
        # We hold the raw images to draw on later
        raw_images = {}
        
        # 1. Gather Data from all cameras
        for prefix, filepath in cam_files.items():
            if prefix not in PREFIX_TO_INDEX: continue
            cam_idx = PREFIX_TO_INDEX[prefix]
            
            img = cv2.imread(filepath)
            raw_images[cam_idx] = img
            
            # Get GT
            T_gt = get_aruco_pose_world(img, cam_idx, intrinsic_mats, world_to_cam_mats, dist_coeffs)
            if T_gt is not None: gt_poses_world.append(T_gt)
            
            # Get Pred
            pred = get_model_prediction_world(img, cam_idx, intrinsic_mats, world_to_cam_mats)
            if pred is not None: pred_dicts_world.append(pred)

        # 2. Fuse GT
        if not gt_poses_world: continue
        T_GT_FUSED = np.eye(4)
        T_GT_FUSED[:3, 3] = np.mean([t[:3, 3] for t in gt_poses_world], axis=0)
        qs = [R.from_matrix(t[:3, :3]).as_quat() for t in gt_poses_world]
        base_q = qs[0]
        for k in range(1, len(qs)): 
            if np.dot(base_q, qs[k]) < 0: qs[k] *= -1
        T_GT_FUSED[:3, :3] = R.from_quat(np.mean(qs, axis=0)/np.linalg.norm(np.mean(qs, axis=0))).as_matrix()

        # 3. Fuse Predictions
        if not pred_dicts_world: continue
        avg_tvec = np.mean([d['tvec_world'] for d in pred_dicts_world], axis=0)
        rvecs = [d['rvec_world'].flatten() for d in pred_dicts_world]
        avg_rot = R.from_rotvec(rvecs).mean().as_rotvec()
        
        T_PRED_FUSED = np.eye(4)
        R_pred, _ = cv2.Rodrigues(avg_rot)
        T_PRED_FUSED[:3, :3] = R_pred
        T_PRED_FUSED[:3, 3] = avg_tvec

        # 4. Metric
        err_t, err_r = calculate_error(T_GT_FUSED, T_PRED_FUSED)
        results.append({"id": img_id, "err_t_m": float(err_t), "err_r_deg": float(err_r)})
        print(f"ID {img_id}: Used {len(pred_dicts_world)} cams | Err T: {err_t*100:.2f}cm")

        # 5. VISUALIZATION (Project Fused Result back to ALL Cameras)
        vis_imgs = []
        
        # Sort indices to keep display order consistent (1, 2, 3)
        sorted_indices = sorted(raw_images.keys()) # [4, 10, 16] usually
        
        for cam_idx in sorted_indices:
            img_vis = raw_images[cam_idx].copy()
            
            # --- Project Fused GT (GREEN) ---
            # World -> Cam
            T_obj_cam_gt = world_to_cam_mats[cam_idx] @ T_GT_FUSED
            rvec_g, _ = cv2.Rodrigues(T_obj_cam_gt[:3, :3])
            tvec_g = T_obj_cam_gt[:3, 3]
            try:
                pts_g, _ = cv2.projectPoints(CUBIOD_VERTS, rvec_g, tvec_g, intrinsic_mats[cam_idx], dist_coeffs[cam_idx])
                draw_cuboid_2d(img_vis, pts_g.reshape(8, 2), color=(0, 255, 0), thickness=2)
            except: pass

            # --- Project Fused Pred (RED) ---
            T_obj_cam_pred = world_to_cam_mats[cam_idx] @ T_PRED_FUSED
            rvec_p, _ = cv2.Rodrigues(T_obj_cam_pred[:3, :3])
            tvec_p = T_obj_cam_pred[:3, 3]
            try:
                pts_p, _ = cv2.projectPoints(CUBIOD_VERTS, rvec_p, tvec_p, intrinsic_mats[cam_idx], dist_coeffs[cam_idx])
                draw_cuboid_2d(img_vis, pts_p.reshape(8, 2), color=(0, 0, 255), thickness=2)
            except: pass
            
            # Label
            cv2.putText(img_vis, f"Cam {cam_idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Resize for tiling (optional, keeps file size sane)
            img_vis = cv2.resize(img_vis, (640, 400))
            vis_imgs.append(img_vis)

        # Concatenate Horizontally
        if vis_imgs:
            final_composite = np.hstack(vis_imgs)
            # Add Error Text to the whole image
            cv2.putText(final_composite, f"FUSED ERR: T={err_t*100:.1f}cm R={err_r:.1f}deg", 
                        (20, final_composite.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            cv2.imwrite(os.path.join(OUTPUT_FOLDER, f"fused_{img_id}.jpg"), final_composite)

    with open(stats_file_path, 'w') as f: json.dump(results, f, indent=4)
    if results:
        mean_t = np.mean([r['err_t_m'] for r in results]) * 100
        mean_r = np.mean([r['err_r_deg'] for r in results])
        print(f"\n=== FINAL FUSION RESULTS ===")
        print(f"Mean Translation Error: {mean_t:.2f} cm")
        print(f"Mean Rotation Error:    {mean_r:.2f} deg")

if __name__ == "__main__":
    main()