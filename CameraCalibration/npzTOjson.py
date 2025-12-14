import numpy as np
import json

# --- Configuration ---
# 1. The path to your existing camera file
npz_file_path = 'calibration_d405.npz'

# 2. The path where you want to save the new JSON file
#    This should be datasets/MyDataset/camera.json
json_file_path = 'camera_TEST.json'

# --- Conversion Script ---
print(f"Loading data from: {npz_file_path}")

# Load the NPZ file
data = np.load(npz_file_path)

# IMPORTANT: Find the key for your camera matrix.
# NPZ files have keys for each array. Let's see what they are.
print(f"Keys found in the NPZ file: {list(data.keys())}")
# Common keys might be 'camera_matrix', 'intrinsics', or 'K'.
# Change the key in the line below to match what you see in your file.
camera_matrix_key = 'camera_matrix' # <--- CHANGE THIS IF NEEDED

# Get the 3x3 camera matrix from the loaded data
intrinsic_matrix = data[camera_matrix_key]
print("Found camera matrix:\n", intrinsic_matrix)

# Extract the parameters from the standard matrix positions
# fx is at [0, 0], fy is at [1, 1]
# cx is at [0, 2], cy is at [1, 2]
fx = intrinsic_matrix[0, 0]
fy = intrinsic_matrix[1, 1]
cx = intrinsic_matrix[0, 2]
cy = intrinsic_matrix[1, 2]

# Create the dictionary in the format our new code expects
output_dict = {
    "fx": float(fx),
    "fy": float(fy),
    "cx": float(cx),
    "cy": float(cy)
}

# Write the dictionary to the JSON file
print(f"Saving converted parameters to: {json_file_path}")
with open(json_file_path, 'w') as f:
    json.dump(output_dict, f, indent=4)

print("Conversion complete!")