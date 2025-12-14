import trimesh
import numpy as np
import json
import os

# --- Determine the directory of this script ---
script_directory = os.path.dirname(os.path.abspath(__file__))

# --- Configuration ---
# obj_file_path is now relative to the script's directory
obj_filename = "my_tracked_object.obj"
obj_file_path = os.path.join(script_directory, obj_filename)

output_ply_filename = "my_object_model.ply" # Desired output .ply filename
output_ply_path = os.path.join(script_directory, output_ply_filename)

models_info_filename = "models_info.json"
models_info_output_path = os.path.join(script_directory, models_info_filename)


category_id_str = "1" # Category ID for models_info.json (as a string)
approx_max_real_world_dim_meters = 0.17 # Approximately 17cm

# --- Helper to determine scale factor ---
def determine_scale_factor(max_extent_original_units, target_max_extent_meters):
    scale_factor = 1.0
    assumed_unit = "meters (or already scaled)"

    # Attempt to guess units based on typical modeling scales
    # Millimeters
    if target_max_extent_meters * 800 < max_extent_original_units < target_max_extent_meters * 1200:
        scale_factor = 0.001
        assumed_unit = "millimeters"
    # Centimeters
    elif target_max_extent_meters * 80 < max_extent_original_units < target_max_extent_meters * 120:
        scale_factor = 0.01
        assumed_unit = "centimeters"
    # Already in Meters (or very close)
    elif target_max_extent_meters * 0.8 < max_extent_original_units < target_max_extent_meters * 1.2:
        scale_factor = 1.0 # No scaling needed
        assumed_unit = "meters"
    else:
        print(f"Warning: Could not confidently determine original units from max extent {max_extent_original_units} and target {target_max_extent_meters}m.")
        # Further guessing based on magnitude
        if max_extent_original_units > target_max_extent_meters * 10: # Likely mm if much larger
             scale_factor = 0.001
             assumed_unit = "millimeters (guessed)"
             print("Guessed millimeters due to large extent. Please verify.")
        elif max_extent_original_units < target_max_extent_meters / 10 and max_extent_original_units > 0: # Likely already scaled or very small
             print("Warning: Model extents are very small compared to target. Assuming it's already in meters or there's a unit mismatch.")
             assumed_unit = "unknown (very small or pre-scaled to meters)"
        else:
            print("Defaulting to no scaling (assuming meters). Please verify model units and output sizes.")


    print(f"Original max model extent: {max_extent_original_units}")
    print(f"Target max real-world dimension (for guessing units): {target_max_extent_meters} meters")
    print(f"Assumed original units: {assumed_unit}")
    print(f"Applied scale factor to convert to meters: {scale_factor}")
    return scale_factor

# --- Main script ---
try:
    print(f"Loading OBJ file from: {obj_file_path}")
    # Use trimesh.load() which is more general
    mesh = trimesh.load(obj_file_path, force='mesh', process=True) # CORRECTED
    
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) == 0:
        print("Warning: Loaded object is not a valid Trimesh instance or has no vertices after processing.")
        print("Attempting to load without processing...")
        mesh = trimesh.load(obj_file_path, force='mesh', process=False) # CORRECTED
        if not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) == 0:
            print("Error: Still could not load a valid mesh even without processing.")
            exit()
        else:
            print("Successfully loaded mesh with 'process=False'. Mesh might not be watertight or have all properties.")


except FileNotFoundError:
    print(f"Error: OBJ file not found at {obj_file_path}")
    print("Please ensure 'my_tracked_object.obj' is in the same directory as this script.")
    exit()
except Exception as e:
    print(f"Error loading OBJ file: {e}")
    print("Please ensure the file path is correct and the file is a valid OBJ mesh.")
    exit()

original_extents = mesh.extents
max_original_extent = np.max(original_extents)

scale = determine_scale_factor(max_original_extent, approx_max_real_world_dim_meters)
mesh.apply_scale(scale)
print(f"Scaled mesh extents (meters): {mesh.extents.tolist()}")

center_kp = mesh.centroid.tolist()

min_coords = mesh.bounds[0]
max_coords = mesh.bounds[1]

# Define 3D bounding box corners relative to the new origin (min_coords) after scaling
# c1: min_x, min_y, min_z
# c2: max_x, min_y, min_z
# c3: max_x, max_y, min_z
# c4: min_x, max_y, min_z
# c5: min_x, min_y, max_z
# c6: max_x, min_y, max_z
# c7: max_x, max_y, max_z
# c8: min_x, max_y, max_z
corners_3d = {
    'c1': [min_coords[0], min_coords[1], min_coords[2]],
    'c2': [max_coords[0], min_coords[1], min_coords[2]],
    'c3': [max_coords[0], max_coords[1], min_coords[2]],
    'c4': [min_coords[0], max_coords[1], min_coords[2]],
    'c5': [min_coords[0], min_coords[1], max_coords[2]],
    'c6': [max_coords[0], min_coords[1], max_coords[2]],
    'c7': [max_coords[0], max_coords[1], max_coords[2]],
    'c8': [min_coords[0], max_coords[1], max_coords[2]]
}

diameter = float(np.linalg.norm(mesh.extents))

keypoints_for_json = {"center": [round(x, 6) for x in center_kp]}
for kp_name, kp_coords in corners_3d.items():
    keypoints_for_json[kp_name] = [round(x, 6) for x in kp_coords]

models_info_content = {
    category_id_str: {
        "kp_3d": keypoints_for_json,
        "diameter": round(diameter, 6),
        "min_x": round(min_coords[0], 6),
        "min_y": round(min_coords[1], 6),
        "min_z": round(min_coords[2], 6),
        "max_x": round(max_coords[0], 6),
        "max_y": round(max_coords[1], 6),
        "max_z": round(max_coords[2], 6),
        "size_x": round(mesh.extents[0], 6),
        "size_y": round(mesh.extents[1], 6),
        "size_z": round(mesh.extents[2], 6)
    }
}

# Output directory is now the same as the script's directory
# No need to create it as it already exists

print("\n--- models_info.json content ---")
print(json.dumps(models_info_content, indent=4))

with open(models_info_output_path, 'w') as f:
    json.dump(models_info_content, f, indent=4)
print(f"\nSuccessfully wrote models_info.json to: {models_info_output_path}")

try:
    mesh.export(output_ply_path)
    print(f"\nSuccessfully exported scaled mesh to: {output_ply_path}")
    print("This .ply file is in METERS.")
except Exception as e:
    print(f"\nError exporting PLY file: {e}")

print("\n--- Important Notes ---")
print("1. Verify Keypoint Order: The c1-c8 keypoints are derived from the bounding box corners in a specific order.")
print("   This order is standard for an axis-aligned bounding box (AABB).")
print("   c1: (min_x, min_y, min_z)")
print("   c2: (max_x, min_y, min_z)")
print("   c3: (max_x, max_y, min_z)")
print("   c4: (min_x, max_y, min_z)")
print("   c5: (min_x, min_y, max_z)")
print("   c6: (max_x, min_y, max_z)")
print("   c7: (max_x, max_y, max_z)")
print("   c8: (min_x, max_y, max_z)")
print("   Ensure your 2D keypoint annotations correspond to these 3D definitions if you have a specific convention.")
print("2. Units: The script attempted to scale your model to meters. ")
print(f"   The scaled model has extents (size_x, size_y, size_z) of approximately: {[round(x, 3) for x in mesh.extents]} meters.")
print(f"   The calculated diameter is approximately: {round(diameter, 3)} meters.")
print("   Please check these values. If they seem incorrect (e.g., too large or too small for an object ~17cm),")
print("   the original unit assumption might have been wrong. You may need to:")
print("     a) Manually edit the scale_factor logic in the `determine_scale_factor` function if you know the original units (e.g., 0.001 for mm, 0.01 for cm).")
print("     b) Or, if the output .ply file looks correct in a viewer but `models_info.json` has wrong scale, ")
print("        you might need to manually adjust the coordinate values and diameter in `models_info.json`.")
print("3. Files Created: `models_info.json` and `my_object_model.ply` should now be in the same directory as this script.")