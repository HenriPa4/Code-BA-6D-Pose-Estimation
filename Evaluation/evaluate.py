import os
import cv2
import cv2.aruco as aruco
import numpy as np
import json
import onnxruntime as rt
from scipy.spatial.transform import Rotation as R
from object_pose_utils_onnx import draw_cuboid_2d

# === CONFIGURATION ===
IMAGE_FOLDER = "Pictures"
OUTPUT_FOLDER = "Evaluation_Results"
MODEL_PATH = 'Models/yolox.onnx' 
INPUT_WIDTH, INPUT_HEIGHT = 1280, 800
CONF_THRES = 0.3

# === PHYSICAL DEFINITIONS (METERS) ===
ARUCO_MARKER_SIZE_METERS = 0.18
CUBE_HALF_EXTENT = 0.1
OBJ_DIM_X = 0.71 
OBJ_DIM_Y = 0.69
OBJ_DIM_Z = 1.02 
ID_TOP = 10  # We prioritize this marker ID as the "Anchor"

# Setup Output
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
stats_file_path = os.path.join(OUTPUT_FOLDER, "evaluation_metrics.json")

# === 1. Define 3D Cuboid in METERS ===
def get_cuboid_verts_meters():
    dim_x, dim_y, dim_z = OBJ_DIM_X, OBJ_DIM_Y, OBJ_DIM_Z
    w, h, d = dim_x / 2.0, dim_y / 2.0, dim_z / 2.0
    # Order matches object_pose_utils_onnx order
    vertices = np.array([
        [-w, -h, -d], [-w, -h,  d], [-w,  h,  d], [-w,  h, -d],
        [ w, -h, -d], [ w, -h,  d], [ w,  h,  d], [ w,  h, -d],
    ], dtype=np.float32)
    return vertices

CUBIOD_VERTS_METERS = get_cuboid_verts_meters()

# === 2. ARUCO SETUP (Ground Truth) ===
def create_ref_transform(rot_axis, angle_deg, translation):
    r = R.from_euler(rot_axis, angle_deg, degrees=True)
    mat = np.eye(4)
    mat[:3, :3] = r.as_matrix()
    mat[:3, 3] = translation
    return mat

T_REF_FRONT = create_ref_transform('y', 0,   [0, 0, CUBE_HALF_EXTENT])
T_REF_RIGHT = create_ref_transform('y', 90,  [CUBE_HALF_EXTENT, 0, 0])
T_REF_BACK  = create_ref_transform('y', 180, [0, 0, -CUBE_HALF_EXTENT])
T_REF_LEFT  = create_ref_transform('y', -90, [-CUBE_HALF_EXTENT, 0, 0])
T_REF_TOP   = create_ref_transform('x', -90, [0, -CUBE_HALF_EXTENT, 0])

MARKER_POSES = {11: T_REF_FRONT, 12: T_REF_RIGHT, 13: T_REF_BACK, 14: T_REF_LEFT, 10: T_REF_TOP}

# Robot Offset
rot_matrix = create_ref_transform('zx', [180, -90], [0,0,0])
x_offset = CUBE_HALF_EXTENT + 0.39 - (OBJ_DIM_X / 2.0)
y_offset = CUBE_HALF_EXTENT + 0.27 - (OBJ_DIM_Y / 2.0)
z_offset = - ((OBJ_DIM_Z / 2.0) + 0.05)
trans_matrix = np.eye(4)
trans_matrix[:3, 3] = np.array([x_offset, y_offset, z_offset])
T_REF_OBJECT = rot_matrix @ trans_matrix

def estimatePoseSingleMarkers(corners, marker_size, mtx, distortion):
    marker_points = np.array([[-marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, -marker_size / 2, 0],
                              [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)
    trash, rvecs, tvecs = [], [], []
    for c in corners:
        nada, R_out, t_out = cv2.solvePnP(marker_points, c, mtx, distortion, False, cv2.SOLVEPNP_IPPE_SQUARE)
        rvecs.append(R_out)
        tvecs.append(t_out)
        trash.append(nada)
    return rvecs, tvecs, trash

# === 3. MODEL SETUP (Prediction) ===
session = rt.InferenceSession(MODEL_PATH, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

def convert_6d_to_matrix(d6_rot):
    a1, a2 = d6_rot[:3], d6_rot[3:]
    b1 = a1 / np.linalg.norm(a1)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / np.linalg.norm(b2)
    b3 = np.cross(b1, b2)
    return np.stack((b1, b2, b3), axis=1)

def get_calibration(camera_id):
    calib_file = f"calibration_{camera_id}.npz"
    try:
        data = np.load(calib_file)
        return data['camera_matrix'], data['dist_coeffs']
    except Exception as e:
        print(f"Warning: Could not load {calib_file}: {e}")
        return None, None

def get_aruco_pose(img, cam_matrix, dist_coeffs):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    try: aruco_params = aruco.DetectorParameters()
    except AttributeError: aruco_params = aruco.DetectorParameters_create()
    aruco_params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    
    try: detector = aruco.ArucoDetector(aruco_dict, aruco_params)
    except AttributeError: detector = None
    
    if detector: corners, ids, _ = detector.detectMarkers(gray)
    else: corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
    
    if ids is None: return None
    
    rvecs, tvecs, _ = estimatePoseSingleMarkers(corners, ARUCO_MARKER_SIZE_METERS, cam_matrix, dist_coeffs)
    
    # 1. Collect all candidates
    candidates_list = []
    for i, marker_id in enumerate(ids.flatten()):
        if marker_id in MARKER_POSES:
            rmat, _ = cv2.Rodrigues(rvecs[i])
            tvec = tvecs[i].reshape(3, 1)
            T_cam_marker = np.eye(4)
            T_cam_marker[:3, :3] = rmat
            T_cam_marker[:3, 3] = tvec.flatten()
            
            # Calculate where the Cube Center is based on this marker
            T_cam_ref = T_cam_marker @ np.linalg.inv(MARKER_POSES[marker_id])
            
            candidates_list.append({
                "id": marker_id,
                "T_cam_ref": T_cam_ref
            })
    
    if not candidates_list: return None

    # --- THE FIX: CONSISTENCY CHECK ---
    
    # 2. Sort: Trust ID 10 (Top) the most, put it first
    candidates_list.sort(key=lambda x: 0 if x["id"] == ID_TOP else 1)
    
    valid_transforms = []
    
    # The first one is our "Base" / "Anchor"
    T_ref_base = candidates_list[0]["T_cam_ref"]
    R_ref_base = T_ref_base[:3, :3]
    
    valid_transforms.append(T_ref_base)
    
    # 3. Check others against the Base
    for k in range(1, len(candidates_list)):
        cand = candidates_list[k]
        T_est = cand["T_cam_ref"]
        R_est = T_est[:3, :3]
        
        # Calculate rotation difference trace
        # If trace is near 3, rotations match. If trace < 0.5, significant flip.
        R_diff = R_ref_base.T @ R_est
        trace = np.trace(R_diff)
        
        if trace < 0.5:
            # FLIP DETECTED!
            # Do NOT use this marker's rotation. Use the Base's rotation/transform instead.
            # This mimics the "overwrite" behavior of the annotation script.
            valid_transforms.append(T_ref_base)
        else:
            valid_transforms.append(T_est)

    # 4. Average the (now consistent) transforms
    translations = [t[:3, 3] for t in valid_transforms]
    avg_translation = np.mean(translations, axis=0)
    
    quats = [R.from_matrix(t[:3, :3]).as_quat() for t in valid_transforms]
    base_q = quats[0]
    for k in range(1, len(quats)):
        if np.dot(base_q, quats[k]) < 0: quats[k] *= -1
    avg_quat = np.mean(quats, axis=0)
    avg_quat /= np.linalg.norm(avg_quat)
    
    T_cam_ref_final = np.eye(4)
    T_cam_ref_final[:3, :3] = R.from_quat(avg_quat).as_matrix()
    T_cam_ref_final[:3, 3] = avg_translation
    
    return T_cam_ref_final @ T_REF_OBJECT

def get_model_pose(img_original, cam_matrix):
    img_h, img_w = img_original.shape[:2]
    input_img = cv2.resize(img_original, (INPUT_WIDTH, INPUT_HEIGHT), interpolation=cv2.INTER_LINEAR)
    blob = np.ascontiguousarray(input_img.transpose((2, 0, 1))[None, :, :, :], dtype=np.float32)
    
    outputs = session.run(None, {session.get_inputs()[0].name: blob})
    dets = outputs[0]
    
    if dets.ndim == 3: dets = dets[0]
    if dets.ndim == 1: dets = dets[None, :] 
    
    best_det = None
    best_score = -1
    for det in dets:
        score = det[4]
        if score > CONF_THRES and score > best_score:
            best_score = score
            best_det = det
            
    if best_det is None: return None
    
    rot_6d = best_det[6:12]
    rot_mat = convert_6d_to_matrix(rot_6d)
    tvec = best_det[12:15] # Model output is Meters
    
    T_cam_object_pred = np.eye(4)
    T_cam_object_pred[:3, :3] = rot_mat
    T_cam_object_pred[:3, 3] = tvec
    
    return T_cam_object_pred

def calculate_errors(T_gt, T_pred):
    trans_gt = T_gt[:3, 3]
    trans_pred = T_pred[:3, 3]
    dist_error_meters = np.linalg.norm(trans_gt - trans_pred)
    
    R_gt = T_gt[:3, :3]
    R_pred = T_pred[:3, :3]
    R_diff = R_gt.T @ R_pred
    trace = np.trace(R_diff)
    trace = max(-1.0, min(3.0, trace))
    angle_rad = np.arccos((trace - 1) / 2)
    angle_deg = np.degrees(angle_rad)
    
    return dist_error_meters, angle_deg

def main():
    image_files = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    results = []
    
    print(f"Starting evaluation on {len(image_files)} images...")

    for fname in sorted(image_files):
        path = os.path.join(IMAGE_FOLDER, fname)
        img = cv2.imread(path)
        if img is None: continue
        
        cam_id = fname[0] 
        cam_mat, dist = get_calibration(cam_id)
        if cam_mat is None: continue
        
        T_gt = get_aruco_pose(img, cam_mat, dist)
        T_pred = get_model_pose(img, cam_mat)
        
        status_text = []
        error_t, error_r = None, None
        
        pts_3d_obj = CUBIOD_VERTS_METERS 

        # Draw GT (Green)
        img_gt = img.copy()
        if T_gt is not None:
            rvec_gt, _ = cv2.Rodrigues(T_gt[:3, :3])
            cuboid_2d_gt, _ = cv2.projectPoints(pts_3d_obj, rvec_gt, T_gt[:3, 3], cam_mat, dist)
            draw_cuboid_2d(img_gt, cuboid_2d_gt.reshape(8, 2), color=(0, 255, 0), thickness=3)
            cv2.putText(img_gt, "GT (Aruco)", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
             cv2.putText(img_gt, "GT Not Found", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Draw Pred (Red)
        img_pred = img.copy()
        if T_pred is not None:
            rvec_pred, _ = cv2.Rodrigues(T_pred[:3, :3])
            cuboid_2d_pred, _ = cv2.projectPoints(pts_3d_obj, rvec_pred, T_pred[:3, 3], cam_mat, dist)
            draw_cuboid_2d(img_pred, cuboid_2d_pred.reshape(8, 2), color=(0, 0, 255), thickness=3)
            cv2.putText(img_pred, "Pred (Model)", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
             cv2.putText(img_pred, "Pred Not Found", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        if T_gt is not None and T_pred is not None:
            error_t, error_r = calculate_errors(T_gt, T_pred)
            status_text.append(f"d_T: {error_t*100:.2f}cm")
            status_text.append(f"d_R: {error_r:.2f}deg")
            
            results.append({
                "image": fname,
                "error_translation_m": float(error_t),
                "error_rotation_deg": float(error_r),
                "gt_translation": T_gt[:3, 3].tolist(),
                "pred_translation": T_pred[:3, 3].tolist()
            })
        
        combined = np.hstack((img_gt, img_pred))
        
        if error_t is not None:
            info = f"Err T: {error_t*100:.2f} cm | Err R: {error_r:.2f} deg"
            cv2.putText(combined, info, (combined.shape[1]//2 - 300, combined.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        
        scale = 0.5
        combined_small = cv2.resize(combined, (0,0), fx=scale, fy=scale)
        out_path = os.path.join(OUTPUT_FOLDER, f"eval_{fname}")
        cv2.imwrite(out_path, combined_small)
        print(f"Processed {fname} | {status_text}")

    with open(stats_file_path, 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"\nDone. Results saved to {OUTPUT_FOLDER}")
    if len(results) > 0:
        avg_t = np.mean([r['error_translation_m'] for r in results])
        avg_r = np.mean([r['error_rotation_deg'] for r in results])
        print(f"AVERAGE Translation Error: {avg_t*100:.2f} cm")
        print(f"AVERAGE Rotation Error:    {avg_r:.2f} deg")

if __name__ == "__main__":
    main()