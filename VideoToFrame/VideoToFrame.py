import pyrealsense2 as rs
import numpy as np
import cv2
import os

def bag_to_pngs(bag_file_path, output_folder_path, stream_type="color", bag_index=0):
    """
    Extracts frames from a .bag file and saves them as PNG images.

    Args:
        bag_file_path (str): Path to the .bag file.
        output_folder_path (str): Path to the folder where PNGs will be saved.
        stream_type (str): Type of stream to extract ("color", "depth", "infrared").
                           Note: Depth will be saved as a scaled 8-bit or raw 16-bit image.
    """
    if not os.path.exists(bag_file_path):
        print(f"Error: Bag file not found at {bag_file_path}")
        return

    os.makedirs(output_folder_path, exist_ok=True)
    print(f"Outputting {stream_type} frames to: {output_folder_path}")

    # Configure pipeline to read from a .bag file
    pipeline = rs.pipeline()
    config = rs.config()
    rs.config.enable_device_from_file(config, bag_file_path, repeat_playback=False)

    # Attempt to enable the specified stream
    # You might need to know the exact stream configuration from the bag file
    # This is a common configuration, but bag files can vary.
    if stream_type == "color":
        config.enable_stream(rs.stream.color) # Let SDK auto-select format and resolution
    elif stream_type == "depth":
        config.enable_stream(rs.stream.depth)
    elif stream_type == "infrared":
        config.enable_stream(rs.stream.infrared, 1) # Assuming infrared stream 1
        # Or try config.enable_stream(rs.stream.infrared) for auto-selection
    else:
        print(f"Error: Unsupported stream type '{stream_type}'. Choose 'color', 'depth', or 'infrared'.")
        return

    try:
        # Start streaming from the file
        profile = pipeline.start(config)

        # Get the playback device
        playback = profile.get_device().as_playback()
        if not playback:
            print("Error: Could not get playback device from bag file.")
            return

        # Disable real-time playback to process all frames
        playback.set_real_time(False)

        print("Processing frames...")
        frame_count = 0
        align = None # Initialize align object

        # For depth visualization (optional, but good for saving depth as viewable PNGs)
        if stream_type == "depth":
            # Create an alignment object
            # rs.align allows us to perform alignment of depth frames to others frames
            # "align_to" is the stream type to which we plan to align depth frames.
            align_to = rs.stream.color # Or rs.stream.depth if only depth is present
            align = rs.align(align_to)
            # For saving depth, we might want a colorizer for visual PNGs
            # Or save raw 16-bit depth data
            # colorizer = rs.colorizer()

        while True:
            try:
                # Wait for the next set of frames from the bag file
                # Timeout in milliseconds. Increase if processing is very slow.
                frames = pipeline.wait_for_frames(timeout_ms=5000)
            except RuntimeError as e:
                if "Frames not received within " in str(e):
                    print("No more frames in the bag file or timeout reached.")
                else:
                    print(f"RuntimeError: {e}")
                break # Exit the loop if no frames are received (end of file or error)

            if not frames:
                print("No frames received, exiting.")
                break

            # Align frames if depth is involved and alignment is desired
            if align:
                frames = align.process(frames)

            target_frame = None
            if stream_type == "color":
                target_frame = frames.get_color_frame()
            elif stream_type == "depth":
                target_frame = frames.get_depth_frame()
                # Optional: Colorize depth for visualization
                # if target_frame:
                #     target_frame = colorizer.colorize(target_frame)
            elif stream_type == "infrared":
                target_frame = frames.get_infrared_frame(1) # Or just get_infrared_frame()
                if not target_frame: # Try stream index 0 if 1 fails
                    target_frame = frames.get_infrared_frame(0)


            if target_frame:
                # Get frame data as a numpy array
                image_data = np.asanyarray(target_frame.get_data())

                # change thresholds accordingly!!!!!!!!!!!!!!!!!!!!!!!!!
                i=0
                if(bag_index <= 2):
                    i=1
                elif(bag_index <= 4):
                    i=2
                elif(bag_index <= 6):
                    i=3
                elif(bag_index <= 8):
                    i=4

                filename = os.path.join(output_folder_path, f"{i}_" + f"{bag_index}_" + f"{frame_count:06d}" + ".png")

                if stream_type == "color":
                    # RealSense provides RGB. OpenCV's imwrite expects BGR.
                    image_to_save = cv2.cvtColor(image_data, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(filename, image_to_save)
                elif stream_type == "depth":
                    # If colorized (target_frame = colorizer.colorize(target_frame)), image_data will be 3-channel RGB
                    # if image_data.ndim == 3 and image_data.shape[2] == 3: # Colorized depth
                    #     image_to_save = cv2.cvtColor(image_data, cv2.COLOR_RGB2BGR)
                    #     cv2.imwrite(filename, image_to_save)
                    # else: # Raw depth data (usually 16-bit single channel)
                        # You can save it as is (will be a 16-bit PNG)
                        # Or scale it to 8-bit for easier viewing if desired
                        # depth_scaled_8bit = cv2.convertScaleAbs(image_data, alpha=255.0/image_data.max() if image_data.max() > 0 else 0)
                        # cv2.imwrite(filename, depth_scaled_8bit)
                    cv2.imwrite(filename, image_data) # Saves as 16-bit grayscale if raw depth
                elif stream_type == "infrared":
                    # Infrared is usually 8-bit or 16-bit grayscale
                    cv2.imwrite(filename, image_data)

                if frame_count % 100 == 0: # Print progress every 100 frames
                    print(f"Saved {filename}")
                frame_count += 1
            # else:
                # This can happen if the stream is intermittent or not present in a particular frameset
                # print(f"No {stream_type} frame found in this frameset.")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Stop streaming
        pipeline.stop()
        print(f"\nProcessing complete. Extracted {frame_count} {stream_type} frames.")
        print(f"PNGs saved to: {output_folder_path}")


if __name__ == "__main__":
    INPUT_FILE = "video"
    OUTPUT_FOLDER = "Train"
    STREAM = "color" #color, depth, infrared

    i = 1
    while True:
        if os.path.exists(INPUT_FILE + str(i) + ".bag"):
            bag_to_pngs(INPUT_FILE + str(i) + ".bag", OUTPUT_FOLDER, STREAM, i)
        else:
            break
        i += 1

    THIRD_FOLDER = "Val"
    os.makedirs(THIRD_FOLDER, exist_ok=True)
    images = sorted([f for f in os.listdir(OUTPUT_FOLDER) if f.endswith(".png")])
    for idx, img in enumerate(images):
        if (idx + 1) % 3 == 0:
            src = os.path.join(OUTPUT_FOLDER, img)
            dst = os.path.join(THIRD_FOLDER, img)
            os.rename(src, dst)

    print(f"Moved every 3rd image to: {THIRD_FOLDER}")

    # Example usage (if not using command line):
    # bag_file_path = "path/to/your/recording.bag"
    # output_folder_path = "path/to/your/output_pngs"
    # desired_stream = "color" # or "depth" or "infrared"
    #
    # if os.path.exists(bag_file_path):
    #     bag_to_pngs(bag_file_path, output_folder_path, desired_stream)
    # else:
    #     print(f"Please set a valid 'bag_file_path'. Current: {bag_file_path}")