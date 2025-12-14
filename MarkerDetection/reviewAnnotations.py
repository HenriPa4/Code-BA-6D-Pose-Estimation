import cv2
import os
import glob
import numpy as np

# --- Configuration ---
# Folder containing the debug visualization images with bounding boxes and axes.
DEBUG_FOLDER = "DebugVisualizations"
# Folder containing the original, raw images.
ORIGINAL_IMAGE_FOLDER = "Pictures"

# --- Main Script ---
def review_images():
    """
    A script to review images in the debug folder, navigate with arrow keys,
    and delete bad annotations/images by pressing the 'd' key.
    """
    # Get a sorted list of all debug images.
    debug_image_paths = sorted(glob.glob(os.path.join(DEBUG_FOLDER, "debug_*.png")))
    debug_image_paths += sorted(glob.glob(os.path.join(DEBUG_FOLDER, "debug_*.jpg")))
    debug_image_paths += sorted(glob.glob(os.path.join(DEBUG_FOLDER, "debug_*.jpeg")))


    if not debug_image_paths:
        print(f"No debug images found in '{DEBUG_FOLDER}'. Exiting.")
        return

    print("--- Image Review Tool ---")
    print(f"Found {len(debug_image_paths)} images to review.")
    print("Controls:")
    print("  -> : Next Image")
    print("  <- : Previous Image")
    print("  d  : Delete current image (both debug and original)")
    print("  q  : Quit")
    print("-------------------------")

    current_index = 0
    while True:
        # If the list is empty after deletions, exit gracefully.
        if not debug_image_paths:
            print("\nAll images have been deleted. Exiting.")
            break

        # Ensure the index is always valid, especially after deleting the last image.
        current_index = max(0, min(current_index, len(debug_image_paths) - 1))

        # --- Get file paths ---
        debug_path = debug_image_paths[current_index]
        debug_filename = os.path.basename(debug_path)
        # The original filename is the debug filename without the "debug_" prefix.
        original_filename = debug_filename.replace("debug_", "")
        original_path = os.path.join(ORIGINAL_IMAGE_FOLDER, original_filename)

        # --- Display the image ---
        image = cv2.imread(debug_path)
        if image is None:
            print(f"Warning: Could not read image {debug_path}. Skipping.")
            # Remove the problematic file from the list and continue.
            debug_image_paths.pop(current_index)
            continue

        # Add text to the image to show info
        font = cv2.FONT_HERSHEY_SIMPLEX
        info_text = f"({current_index + 1}/{len(debug_image_paths)}) {original_filename}"
        cv2.putText(image, info_text, (10, 30), font, 1, (0, 0, 0), 3, cv2.LINE_AA) # Black outline
        cv2.putText(image, info_text, (10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA) # White text

        window_name = "Annotation Review"
        cv2.imshow(window_name, image)
        
        # --- Wait for user input ---
        key = cv2.waitKey(0)

        # --- Handle Key Presses ---
        # Quit
        if key == ord('q'):
            print("Exiting review tool.")
            break
        # Next image (Right Arrow)
        # Common key codes for right arrow are 2555904 (Linux/macOS) and 2228224 (some systems)
        elif key == 83:
            current_index = (current_index + 1) % len(debug_image_paths)
        # Previous image (Left Arrow)
        # Common key codes for left arrow are 2424832 (Linux/macOS) and 2162688 (some systems)
        elif key == 81:
            current_index = (current_index - 1 + len(debug_image_paths)) % len(debug_image_paths)
        # Delete image
        elif key == ord('d'):
            print(f"\nDeleting: ")
            print(f"  - {debug_path}")
            print(f"  - {original_path}")

            # Delete the debug image
            try:
                os.remove(debug_path)
            except OSError as e:
                print(f"Error deleting debug file: {e}")

            # Delete the original image
            try:
                os.remove(original_path)
            except OSError as e:
                print(f"Error deleting original file: {e}")
            
            # Remove the deleted path from our navigation list
            debug_image_paths.pop(current_index)
            print("Deletion complete.")


    cv2.destroyAllWindows()

if __name__ == "__main__":
    review_images()