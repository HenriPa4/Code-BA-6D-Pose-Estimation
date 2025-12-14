import cv2
import cv2.aruco as aruco # Make sure this import is used
import numpy as np
import json
import os
import glob
from datetime import datetime
from scipy.spatial.transform import Rotation as R

# --- Configuration ---
IMAGE_FOLDER = "Pictures"
OUTPUT_COCO_FILE = "coco_annotations.json"
OUTPUT_DEBUG_VIS_FOLDER = "DebugVisualizations"
NPZ_CAMERA_MATRIX_KEY = 'camera_matrix'
NPZ_DIST_COEFFS_KEY = 'dist_coeffs'
ARUCO_DICTIONARY_ENUM = aruco.DICT_4X4_50 

# 1. UPDATE THIS: Size of the printed marker
ARUCO_MARKER_SIZE_METERS = 0.18  

# 2. UPDATE THIS: Distance from Cube Center to Marker Surface 
# (e.g., if cube is 10cm wide, this is 0.1)
CUBE_HALF_EXTENT = 0.1 

# 3. UPDATE THIS: Object Physical Dimensions
OBJECT_NAME = "robot"
OBJECT_CATEGORY_ID = 0
OBJ_DIM_X = 0.71 #x is direction of needle
OBJ_DIM_Y = 0.69 #y is direction of handle
OBJ_DIM_Z = 1.02 #z is direction of TOP

# 4. DEFINE THE CUBE FACES
# Map your specific Marker IDs to the cube faces.
# "Front" is usually the face looking at the camera when the object is upright.
# Adjust these IDs to match your physical cube.
ID_TOP   = 10
ID_FRONT = 11 #handle
ID_RIGHT = 12 
ID_BACK  = 13
ID_LEFT  = 14 #needle

# Helper to create 4x4 matrix from rotation vector (degrees) and translation
def create_ref_transform(rot_axis, angle_deg, translation):
    r = R.from_euler(rot_axis, angle_deg, degrees=True)
    mat = np.eye(4)
    mat[:3, :3] = r.as_matrix()
    mat[:3, 3] = translation
    return mat

# --- DEFINE MARKER POSES RELATIVE TO CUBE CENTER ---
# Coordinate System:
# Ref Frame = Center of Cube.
# X+ = Right, Y+ = Down, Z+ = Forward (Standard OpenCV)
# Markers define their own Z as coming "out" of the marker.

# Front Face (Z+ direction relative to center)
T_REF_FRONT = create_ref_transform('y', 0,   [0, 0, CUBE_HALF_EXTENT])

# Right Face (X+ direction) - Rotated 90 deg around Y
T_REF_RIGHT = create_ref_transform('y', 90,  [CUBE_HALF_EXTENT, 0, 0])

# Back Face (Z- direction) - Rotated 180 deg around Y
T_REF_BACK  = create_ref_transform('y', 180, [0, 0, -CUBE_HALF_EXTENT])

# Left Face (X- direction) - Rotated -90 deg around Y
T_REF_LEFT  = create_ref_transform('y', -90, [-CUBE_HALF_EXTENT, 0, 0])

# Top Face (Y- direction) - Rotated -90 deg around X
T_REF_TOP   = create_ref_transform('x', -90, [0, -CUBE_HALF_EXTENT, 0])

MARKER_POSES_IN_REFERENCE_FRAME = {
    ID_FRONT: T_REF_FRONT,
    ID_RIGHT: T_REF_RIGHT,
    ID_BACK:  T_REF_BACK,
    ID_LEFT:  T_REF_LEFT,
    ID_TOP:   T_REF_TOP,
}

#transform into robot coordinate system
rot_matrix = create_ref_transform('zx', [180, -90], [0,0,0])

# Calculate the axis offsets
# Vector from cube center to robot center
x_offset = CUBE_HALF_EXTENT + 0.39 - (OBJ_DIM_X / 2.0)
y_offset = CUBE_HALF_EXTENT + 0.27 - (OBJ_DIM_Y / 2.0)
z_offset = - ((OBJ_DIM_Z / 2.0) + 0.15)

t_obj_in_ref = np.array([x_offset, y_offset, z_offset])

trans_matrix = np.eye(4)
trans_matrix[:3, 3] = np.array([x_offset, y_offset, z_offset])

# C. Combine: ROTATE first, THEN TRANSLATE
T_REF_OBJECT = rot_matrix @ trans_matrix


# --- Helper Functions (Keep all of them: get_object_3d_model_points, rvec_tvec_to_matrix, etc.) ---
def get_object_3d_model_points(dim_x, dim_y, dim_z):
    w, h, d = dim_x / 2.0, dim_y / 2.0, dim_z / 2.0
    center = np.array([[0, 0, 0]], dtype=np.float32)
    corners = np.array([
        [-w, -h, -d], [w, -h, -d], [w, h, -d], [-w, h, -d],
        [-w, -h,  d], [w, -h,  d], [w, h,  d], [-w, h,  d]
    ], dtype=np.float32)
    return np.vstack((center, corners))

def rvec_tvec_to_matrix(rvec, tvec):
    R_mat, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R_mat
    T[:3, 3] = tvec.flatten()
    return T

def matrix_to_rvec_tvec(T):
    R_mat = T[:3, :3]
    tvec = T[:3, 3].reshape(3, 1)
    rvec, _ = cv2.Rodrigues(R_mat)
    return rvec, tvec

def draw_projected_bbox3d(image, points_2d):
    if points_2d is None or len(points_2d) == 0: return image
    points_2d_int = np.int32(points_2d.reshape(-1, 2))
    if len(points_2d_int) > 0:
        cv2.circle(image, tuple(points_2d_int[0]), 5, (255, 0, 255), -1)
    edges = [
        (1, 2), (2, 3), (3, 4), (4, 1), (5, 6), (6, 7), (7, 8), (8, 5),
        (1, 5), (2, 6), (3, 7), (4, 8)
    ]
    for i, j in edges:
        if i < len(points_2d_int) and j < len(points_2d_int):
             cv2.line(image, tuple(points_2d_int[i]), tuple(points_2d_int[j]), (0, 255, 0), 2)
    return image

def robust_average_quaternions(quaternions_list):
    if not quaternions_list:
        return R.from_quat([0, 0, 0, 1])
    qs = np.array([q.as_quat() for q in quaternions_list])
    for i in range(1, len(qs)):
        if np.dot(qs[0], qs[i]) < 0:
            qs[i] *= -1
    q_avg_raw = np.mean(qs, axis=0)
    q_avg_normalized = q_avg_raw / np.linalg.norm(q_avg_raw)
    return R.from_quat(q_avg_normalized)

def estimatePoseSingleMarkers(corners, marker_size, mtx, distortion):
    '''
    This will estimate the rvec and tvec for each of the marker corners detected by:
       corners, ids, rejectedImgPoints = detector.detectMarkers(image)
    corners - is an array of detected corners for each detected marker in the image
    marker_size - is the size of the detected markers
    mtx - is the camera matrix
    distortion - is the camera distortion matrix
    RETURN list of rvecs, tvecs, and trash (so that it corresponds to the old estimatePoseSingleMarkers())
    '''
    marker_points = np.array([[-marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, -marker_size / 2, 0],
                              [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)
    trash = []
    rvecs = []
    tvecs = []
    for c in corners:
        nada, R, t = cv2.solvePnP(marker_points, c, mtx, distortion, False, cv2.SOLVEPNP_IPPE_SQUARE)
        rvecs.append(R)
        tvecs.append(t)
        trash.append(nada)
    return rvecs, tvecs, trash

# --- Main Script ---
def main():
    aruco_dict_obj = aruco.getPredefinedDictionary(ARUCO_DICTIONARY_ENUM) # Use aruco alias
    # Try this first for DetectorParameters:
    try:
        aruco_params_obj = aruco.DetectorParameters() # Use aruco alias
    except AttributeError:
        # Fallback for older OpenCV versions
        print("Note: cv2.aruco.DetectorParameters() not found, trying cv2.aruco.DetectorParameters_create()")
        aruco_params_obj = aruco.DetectorParameters_create() # Use aruco alias

    aruco_params_obj.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX

    detector = aruco.ArucoDetector(aruco_dict_obj, aruco_params_obj) # Use aruco alias

    coco_output = {
        "info": {"description": f"COCO dataset for {OBJECT_NAME} (quat_avg_standalone ref)",
                 "version": "1.0", "year": datetime.now().year,
                 "date_created": datetime.now().isoformat()},
        "licenses": [], "images": [], "annotations": [],
        "categories": [{"id": OBJECT_CATEGORY_ID, "name": OBJECT_NAME, "supercategory": "object",
                        "keypoints": ["center", "c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8"], "skeleton": []}]
    }
    image_id_counter = 0
    annotation_id_counter = 0
    os.makedirs(OUTPUT_DEBUG_VIS_FOLDER, exist_ok=True)
    image_files = sorted(glob.glob(os.path.join(IMAGE_FOLDER, "*.jpg")) + \
                   glob.glob(os.path.join(IMAGE_FOLDER, "*.png")) + \
                   glob.glob(os.path.join(IMAGE_FOLDER, "*.jpeg")))
    object_3d_points_local = get_object_3d_model_points(OBJ_DIM_X, OBJ_DIM_Y, OBJ_DIM_Z)

    successfully_annotated_files = set()
    calibration_cache = {} # To cache loaded calibration files

    for image_path in image_files:
        print(f"Processing: {image_path}")

        # --- Dynamically load calibration file based on image name ---
        image_basename = os.path.basename(image_path)
        camera_id = image_basename[0]

        if camera_id in calibration_cache:
            camera_matrix = calibration_cache[camera_id][NPZ_CAMERA_MATRIX_KEY]
            dist_coeffs = calibration_cache[camera_id][NPZ_DIST_COEFFS_KEY]
        else:
            calibration_file_name = f"calibration_{camera_id}.npz"
            try:
                print(f"-> Loading calibration for camera '{camera_id}' from '{calibration_file_name}'")
                calibration_data = np.load(calibration_file_name)
                camera_matrix = calibration_data[NPZ_CAMERA_MATRIX_KEY]
                dist_coeffs = calibration_data[NPZ_DIST_COEFFS_KEY]
                calibration_cache[camera_id] = {
                    NPZ_CAMERA_MATRIX_KEY: camera_matrix,
                    NPZ_DIST_COEFFS_KEY: dist_coeffs
                }
            except FileNotFoundError:
                print(f"  [ERROR] Calibration file '{calibration_file_name}' not found for image '{image_path}'. Skipping.")
                continue
            except Exception as e:
                print(f"  [ERROR] Failed to load calibration from '{calibration_file_name}': {e}. Skipping.")
                continue


        img_original = cv2.imread(image_path)
        if img_original is None: continue
        debug_img_display = img_original.copy()
        height, width = debug_img_display.shape[:2]
        gray = cv2.cvtColor(debug_img_display, cv2.COLOR_BGR2GRAY)
        image_id_counter += 1
        coco_output["images"].append({
            "id": image_id_counter,
            "width": width,
            "height": height,
            "file_name": image_basename,
            "image_folder": camera_id,  # Add this key with a default scene ID
            "license": 0,
            "date_captured": datetime.now().isoformat()
        })

        corners, ids, rejected = detector.detectMarkers(gray)
        rvec_object_cam, tvec_object_cam = None, None
        T_cam_ref_final = None

        if ids is not None and len(ids) > 0:
            aruco.drawDetectedMarkers(debug_img_display, corners, ids) # Use aruco alias
            estimated_ref_quaternions = []
            estimated_ref_translations = []

            rvecs_all, tvecs_all, _objPoints_markers = estimatePoseSingleMarkers(
                corners, ARUCO_MARKER_SIZE_METERS, camera_matrix, dist_coeffs
            )

            candidates = []

            for i in range(len(ids)):
                marker_id = ids[i][0]
                if marker_id in MARKER_POSES_IN_REFERENCE_FRAME:
                    curr_rvec = rvecs_all[i]
                    curr_tvec = tvecs_all[i]
                    
                    # Calculate Cube Pose from this marker
                    T_cam_marker = rvec_tvec_to_matrix(curr_rvec, curr_tvec)
                    T_ref_marker = MARKER_POSES_IN_REFERENCE_FRAME[marker_id]
                    
                    try:
                        T_inv_ref_marker = np.linalg.inv(T_ref_marker)
                        T_cam_ref_est = T_cam_marker @ T_inv_ref_marker
                        candidates.append({
                            "idx": i,
                            "id": marker_id,
                            "rvec": curr_rvec,
                            "tvec": curr_tvec,
                            "T_cam_ref": T_cam_ref_est
                        })
                    except np.linalg.LinAlgError:
                        continue

            if not candidates:
                continue

            # --- PASS 2: Consistency Check & Flip ---
            # We assume the majority of markers are correct. 
            # We pick a "Reference" pose. The simple way is to take the first one, 
            # but a robust way is to check which pose is most common.
            # For simplicity: Assume candidate[0] is correct (or ID 10/Top is best if available).
            
            # Sort so that if ID 10 (Top) is present, we trust it most (usually most stable)
            candidates.sort(key=lambda x: 0 if x["id"] == ID_TOP else 1)
            
            valid_ref_quaternions = []
            valid_ref_translations = []
            
            # The trusted reference cube rotation
            T_ref_base = candidates[0]["T_cam_ref"] 
            R_ref_base = T_ref_base[:3, :3]

            for cand in candidates:
                curr_rvec = cand["rvec"]
                curr_tvec = cand["tvec"]
                T_est = cand["T_cam_ref"]
                R_est = T_est[:3, :3]

                # Compare this marker's estimated Cube Rotation with the Base Rotation
                # R_diff = R_base^T * R_est. Trace(R_diff) should be close to 3 for Identity.
                R_diff = np.dot(R_ref_base.T, R_est)
                trace = np.trace(R_diff)
                
                # If trace is small (or negative), the rotation is huge (~180 deg) -> FLIP DETECTED
                # A perfect match has trace 3.0. A 90 deg diff has trace 1.0. 
                # A 180 deg diff has trace -1.0.
                if trace < 0.5: 
                    print(f"  [FIX] Marker {cand['id']} disagrees with consensus. OVERWRITING.")
                    
                    # ---------------------------------------------------------
                    # STRATEGY: IGNORE THE BAD MARKER, CALCULATE WHERE IT SHOULD BE
                    # ---------------------------------------------------------
                    
                    # 1. Trust the consensus pose (derived from Marker 10, etc.)
                    # T_ref_base is the calculated position of the CUBE CENTER
                    T_cam_ref_correct = T_ref_base
                    
                    # 2. Look up where Marker 14 SHOULD be on the cube
                    T_ref_marker_config = MARKER_POSES_IN_REFERENCE_FRAME[cand["id"]]
                    
                    # 3. Calculate the perfect Camera->Marker transform
                    # Camera -> Cube Center -> Marker Surface
                    T_cam_marker_corrected = T_cam_ref_correct @ T_ref_marker_config
                    
                    # 4. Extract the perfect rotation and translation
                    curr_rvec, curr_tvec = matrix_to_rvec_tvec(T_cam_marker_corrected)
                    
                    # 5. Overwrite the bad detection in the list
                    # This ensures the drawn axis is perfect (Red, Green, AND Blue)
                    rvecs_all[cand["idx"]] = curr_rvec
                    tvecs_all[cand["idx"]] = curr_tvec
                    
                    # 6. Update the candidate data for the averaging step
                    T_est = T_cam_ref_correct

                # Add to averaging list
                valid_ref_quaternions.append(R.from_matrix(T_est[:3, :3]))
                valid_ref_translations.append(T_est[:3, 3])

                # Draw the (potentially corrected) axis
                cv2.drawFrameAxes(debug_img_display, camera_matrix, dist_coeffs,
                                  curr_rvec, curr_tvec, ARUCO_MARKER_SIZE_METERS * 0.75, thickness=1)
                cv2.putText(debug_img_display, f"ID:{cand['id']}", tuple(np.int32(corners[cand['idx']][0][0])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # --- AVERAGING (Existing Logic) ---
            if valid_ref_quaternions:
                avg_rotation_R = robust_average_quaternions(valid_ref_quaternions)
                # ... rest of your averaging code ...
                avg_rot_matrix = avg_rotation_R.as_matrix()
                avg_translation = np.mean(np.array(valid_ref_translations), axis=0)
                T_cam_ref_final = np.eye(4)
                T_cam_ref_final[:3, :3] = avg_rot_matrix
                T_cam_ref_final[:3, 3] = avg_translation
                
                # Draw the final Consensus Cube Axis
                rvec_final, tvec_final = matrix_to_rvec_tvec(T_cam_ref_final)
                cv2.drawFrameAxes(debug_img_display, camera_matrix, dist_coeffs,
                                  rvec_final, tvec_final, 0.1, thickness=3)

        if T_cam_ref_final is not None:
            T_cam_object = T_cam_ref_final @ T_REF_OBJECT
            rvec_object_cam, tvec_object_cam = matrix_to_rvec_tvec(T_cam_object)
            projected_points_2d_obj, _ = cv2.projectPoints(object_3d_points_local,
                                                           rvec_object_cam, tvec_object_cam,
                                                           camera_matrix, dist_coeffs)
            bbox_x, bbox_y, bbox_w, bbox_h = 0,0,0,0
            if projected_points_2d_obj is not None:
                annotation_id_counter += 1
                keypoints_coco = []
                all_points_x, all_points_y = [], []
                for pt_2d_arr in projected_points_2d_obj:
                    pt_2d = pt_2d_arr.ravel()
                    x_coord, y_coord = int(round(pt_2d[0])), int(round(pt_2d[1]))
                    keypoints_coco.extend([x_coord, y_coord, 2])
                    all_points_x.append(x_coord); all_points_y.append(y_coord)
                corner_points_x = all_points_x[1:]; corner_points_y = all_points_y[1:]
                if corner_points_x and corner_points_y:
                    min_x,max_x=np.min(corner_points_x),np.max(corner_points_x)
                    min_y,max_y=np.min(corner_points_y),np.max(corner_points_y)
                    bbox_x,bbox_y=int(min_x),int(min_y)
                    bbox_w,bbox_h=int(max_x-min_x),int(max_y-min_y)
                    bbox_x=max(0,min(bbox_x,width-1));bbox_y=max(0,min(bbox_y,height-1))
                    bbox_w=max(0,min(bbox_w,width-bbox_x));bbox_h=max(0,min(bbox_h,height-bbox_y))
                    if bbox_w > 0 and bbox_h > 0:
                        area=float(bbox_w*bbox_h)
                        num_keypoints=len(object_3d_points_local)
                        # --- THIS IS THE NEW, BETTER CODE for quarternion.py ---
                        # Convert the rvec to a 3x3 rotation matrix
                        R_mat, _ = cv2.Rodrigues(rvec_object_cam)

                        # Create the standard 3x4 [R|t] pose matrix
                        pose_matrix = np.hstack((R_mat, tvec_object_cam))

                        coco_output["annotations"].append({
                            "id": annotation_id_counter,
                            "image_id": image_id_counter,
                            "category_id": OBJECT_CATEGORY_ID,
                            "segmentation": [],
                            "area": area,
                            "bbox": [bbox_x, bbox_y, bbox_w, bbox_h],
                            "iscrowd": 0,
                            "keypoints": keypoints_coco,
                            "num_keypoints": num_keypoints,
                            # Use the standard "pose" key and save the 3x4 matrix
                            "R": R_mat.flatten().tolist(),  # Save the 3x3 rotation matrix as a flat list of 9 numbers
                            "T": tvec_object_cam.flatten().tolist() # Save the 3x1 translation vector as a flat list of 3 numbers
                        })

                        successfully_annotated_files.add(os.path.basename(image_path))
        if rvec_object_cam is not None and projected_points_2d_obj is not None:
            cv2.drawFrameAxes(debug_img_display,camera_matrix,dist_coeffs,rvec_object_cam,tvec_object_cam,OBJ_DIM_Z*0.75,thickness=2)
            debug_img_display=draw_projected_bbox3d(debug_img_display,projected_points_2d_obj)
            if bbox_w > 0 and bbox_h > 0:cv2.rectangle(debug_img_display,(bbox_x,bbox_y),(bbox_x+bbox_w,bbox_y+bbox_h),(255,0,0),2)
        display_scale_percent=50
        disp_width=int(debug_img_display.shape[1]*display_scale_percent/100)
        disp_height=int(debug_img_display.shape[0]*display_scale_percent/100)
        #if disp_width>0 and disp_height>0:
        #    resized_debug_img=cv2.resize(debug_img_display,(disp_width,disp_height),interpolation=cv2.INTER_AREA)
        #    cv2.imshow(f"AO - {os.path.basename(image_path)}",resized_debug_img)
        #else:cv2.imshow(f"AO - {os.path.basename(image_path)}",debug_img_display)
        debug_image_filename=f"debug_{os.path.basename(image_path)}"
        debug_output_path=os.path.join(OUTPUT_DEBUG_VIS_FOLDER,debug_image_filename)
        cv2.imwrite(debug_output_path,debug_img_display)
        #key=cv2.waitKey(1)&0xFF
        #if key==ord('q'):print("Exiting...");break
        #if image_id_counter % 10 == 0:
        #    print(f"INFO: Image {image_id_counter} processed. Closing all debug windows (batch of 10 completed).")
        #    cv2.destroyAllWindows()
    cv2.destroyAllWindows()
    with open(OUTPUT_COCO_FILE,'w')as f:json.dump(coco_output,f,indent=4)
    print(f"\nCOCO saved to: {OUTPUT_COCO_FILE}")
    print(f"Debug visualizations saved to: {OUTPUT_DEBUG_VIS_FOLDER}")
    print(f"Total images processed: {image_id_counter}")
    print(f"Total valid annotations: {len(coco_output['annotations'])}")

    print("\n--- Annotation Analysis ---")
    print(f"Total images processed: {len(image_files)}")
    print(f"Total valid annotations created: {len(successfully_annotated_files)}")

    all_processed_basenames = {os.path.basename(f) for f in image_files}
    failed_files = all_processed_basenames - successfully_annotated_files

    if not failed_files:
        print("\nSUCCESS: All images were annotated successfully!")
    else:
        print(f"\nWARNING: {len(failed_files)} image(s) could not be annotated and should be reviewed or removed:")
        for filename in sorted(list(failed_files)):
            print(f"  - {filename}")
            os.remove(os.path.join(IMAGE_FOLDER, f"{filename}"))

if __name__ == "__main__":
    main()