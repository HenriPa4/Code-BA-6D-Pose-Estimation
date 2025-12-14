import cv2
import json
import os
import numpy as np
import glob

# --- Configuration ---
ANNOTATION_FILE = "coco_annotations.json"  # Your COCO JSON file
IMAGE_BASE_FOLDER = "Pictures"             # Folder where original images are stored
OUTPUT_VIS_FOLDER = "OutputVisualize"       # Folder to save images with drawn annotations

# Visualization Colors
BBOX_COLOR = (255, 0, 0)   # Blue for 2D Bounding Box (BGR)
KEYPOINT_COLOR = (0, 255, 0) # Green for Keypoints
CENTER_KEYPOINT_COLOR = (255, 0, 255) # Magenta for the center keypoint (if first one is center)
SKELETON_COLOR = (0, 255, 255) # Yellow for skeleton lines connecting keypoints

# Keypoint skeleton connections (assuming 9 keypoints: center, then 8 corners in a specific order)
# This order should match how your 'get_object_3d_model_points' defined them.
# Example order: center, BRR, BLR, TRR, TLR, BRF, BLF, TRF, TLF (relative to some view)
# The indices below are for the keypoints *after* the center point (i.e., for the 8 corners).
# So, keypoint index 1 in the annotation corresponds to corner_indices[0] here.
# This is a common cuboid skeleton:
# Edges of the 3D bbox (example for a common cuboid connection from your previous script)
# Using 0-based indexing for the 8 corners (after skipping the center point if it exists)
# Original model point indices:
#   0: center
#   1: (-x/2, -y/2, -z/2)  (bottom-left-rear) -> let's call this C0
#   2: ( x/2, -y/2, -z/2)  (bottom-right-rear) -> C1
#   3: ( x/2,  y/2, -z/2)  (top-right-rear)    -> C2
#   4: (-x/2,  y/2, -z/2)  (top-left-rear)     -> C3
#   5: (-x/2, -y/2,  z/2)  (bottom-left-front) -> C4
#   6: ( x/2, -y/2,  z/2)  (bottom-right-front)-> C5
#   7: ( x/2,  y/2,  z/2)  (top-right-front)   -> C6
#   8: (-x/2,  y/2,  z/2)  (top-left-front)    -> C7

# These indices map to the 8 corners if the first keypoint is the center.
# If you only have 8 keypoints (no center), adjust these to be 0-7.
CORNER_SKELETON_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face (e.g., C0-C1, C1-C2, C2-C3, C3-C0)
    (4, 5), (5, 6), (6, 7), (7, 4),  # Top face (e.g., C4-C5, C5-C6, C6-C7, C7-C4)
    (0, 4), (1, 5), (2, 6), (3, 7)   # Connecting edges (e.g., C0-C4, C1-C5, C2-C6, C3-C7)
]
HAS_CENTER_KEYPOINT = True # Set to False if your keypoints are only the 8 corners

# --- Main Script ---
def main():
    # Load COCO annotations
    try:
        with open(ANNOTATION_FILE, 'r') as f:
            coco_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Annotation file not found at {ANNOTATION_FILE}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {ANNOTATION_FILE}")
        return

    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_VIS_FOLDER, exist_ok=True)

    # For quick lookup: map image_id to image_info and annotations
    image_id_to_info = {img_info['id']: img_info for img_info in coco_data['images']}
    
    # Map image_id to a list of its annotations
    image_id_to_annotations = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in image_id_to_annotations:
            image_id_to_annotations[img_id] = []
        image_id_to_annotations[img_id].append(ann)

    print(f"Found {len(image_id_to_info)} images and {len(coco_data['annotations'])} annotations.")

    # Process each image defined in the COCO file
    for image_id, image_info in image_id_to_info.items():
        image_filename = image_info['file_name']
        image_path = os.path.join(IMAGE_BASE_FOLDER, image_filename)

        if not os.path.exists(image_path):
            print(f"Warning: Image file not found: {image_path}. Skipping.")
            continue

        print(f"Processing: {image_path}")
        img = cv2.imread(image_path)
        if img is None:
            print(f"Warning: Could not read image {image_path}. Skipping.")
            continue

        # Get all annotations for this image
        annotations_for_image = image_id_to_annotations.get(image_id, [])

        for ann in annotations_for_image:
            # 1. Draw 2D Bounding Box
            bbox = ann.get('bbox') # [x, y, width, height]
            if bbox:
                x, y, w, h = map(int, bbox)
                cv2.rectangle(img, (x, y), (x + w, y + h), BBOX_COLOR, 2)

            # 2. Draw Keypoints and Skeleton
            keypoints_coco = ann.get('keypoints') # [x1,y1,v1, x2,y2,v2, ...]
            if keypoints_coco:
                num_keypoints_annotated = ann.get('num_keypoints', 0)
                
                parsed_keypoints = [] # Store (x,y) tuples for drawing skeleton
                for kpt_idx in range(0, len(keypoints_coco), 3):
                    x = int(keypoints_coco[kpt_idx])
                    y = int(keypoints_coco[kpt_idx+1])
                    v = int(keypoints_coco[kpt_idx+2]) # Visibility

                    if v > 0: # Only draw if visible or labeled
                        current_color = KEYPOINT_COLOR
                        if HAS_CENTER_KEYPOINT and kpt_idx == 0: # First keypoint is center
                            current_color = CENTER_KEYPOINT_COLOR
                        
                        cv2.circle(img, (x, y), 5, current_color, -1) # Draw filled circle
                        parsed_keypoints.append((x,y))
                    else:
                        parsed_keypoints.append(None) # Placeholder for non-visible/non-labeled

                # Draw skeleton if we have enough keypoints
                actual_keypoints_for_skeleton = []
                if HAS_CENTER_KEYPOINT:
                    if len(parsed_keypoints) > 1:
                        actual_keypoints_for_skeleton = parsed_keypoints[1:] # Skip center for skeleton
                else:
                    actual_keypoints_for_skeleton = parsed_keypoints

                if len(actual_keypoints_for_skeleton) == 8: # Expecting 8 corners for the skeleton
                    for (start_idx, end_idx) in CORNER_SKELETON_EDGES:
                        # Ensure indices are valid and points exist
                        if start_idx < len(actual_keypoints_for_skeleton) and \
                           end_idx < len(actual_keypoints_for_skeleton) and \
                           actual_keypoints_for_skeleton[start_idx] is not None and \
                           actual_keypoints_for_skeleton[end_idx] is not None:
                            
                            pt1 = actual_keypoints_for_skeleton[start_idx]
                            pt2 = actual_keypoints_for_skeleton[end_idx]
                            cv2.line(img, pt1, pt2, SKELETON_COLOR, 1)
            
            # Optional: Draw category name
            category_id = ann.get('category_id')
            if category_id is not None and bbox:
                category_name = "Unknown"
                for cat in coco_data['categories']:
                    if cat['id'] == category_id:
                        category_name = cat['name']
                        break
                cv2.putText(img, category_name, (int(bbox[0]), int(bbox[1])-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, BBOX_COLOR, 2)


        # Save the visualized image
        output_path = os.path.join(OUTPUT_VIS_FOLDER, image_filename)
        cv2.imwrite(output_path, img)
        # print(f"  Saved to: {output_path}")

    print(f"\nVisualization complete. Images saved to: {OUTPUT_VIS_FOLDER}")

if __name__ == "__main__":
    main()