# config.py

# Camera streaming resolution (actual capture)
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 800
CAMERA_FPS = 30

# Display resolution (for live preview)
DISPLAY_WIDTH = 960
DISPLAY_HEIGHT = 540

# Mid ChArUco board parameters (in meters)
#SQUARES_X = 6
#SQUARES_Y = 4
#SQUARE_LENGTH = 0.0891   # e.g. 8.92 cm
#MARKER_LENGTH = 0.069    # e.g. 6.9 cm

# Large ChArUco board parameters (in meters)
SQUARES_X = 4
SQUARES_Y = 3
SQUARE_LENGTH = 0.18   # e.g. 18 cm
MARKER_LENGTH = 0.135    # e.g. 13.5 cm


# ArUco dictionary (choose from OpenCV's predefined dictionaries)
ARUCO_DICT = "DICT_4X4_50"  # We'll map this string to cv2.aruco dictionary

# Burst capture settings
BURST_SIZE = 5

# Timer settings for intrinsic calibration
INTRINSIC_INITIAL_DELAY = 7      # seconds before first capture
INTRINSIC_CAPTURE_INTERVAL = 5      # seconds between subsequent captures
INTRINSIC_NUM_IMAGES = 12           # total number of images to capture

# Directory settings
INTRINSIC_DIR = "images_intrinsic_calibration"
EXTRINSIC_DIR = "images_extrinsic_calibration"
VERIFICATION_DIR = "calibration_verification"

# Data Capture Settings (for capture.py) 
# The top-level directory where all session data will be saved.
CAPTURE_BASE_DIR = "capture_data"

# The target frames-per-second for capturing data.
CAPTURE_FPS = 5

# The display resolution for the small live preview windows.
# A smaller size is better to not clutter the screen.
PREVIEW_WIDTH = 426
PREVIEW_HEIGHT = 240

# Timeout in milliseconds to wait for a coherent frameset from a camera.
FRAME_TIMEOUT_MS = 1000
