import os
import cv2
import re
import numpy as np
import sys
import mediapipe as mp # Import MediaPipe for hand detection

def collect_and_process_images(base_images_path):
    """
    Performs image preprocessing and data collection, including MediaPipe hand
    detection and cropping, for training a sign language recognition model.
    Images are returned with pixel values in the [0, 255] range (as float32),
    as normalization to [-1, 1] will be handled by the MobileNetV2 preprocessing layer in the trainer.

    Args:
        base_images_path (str): The base directory where image subfolders are located.

    Returns:
        tuple: A tuple containing:
            - X (np.array): NumPy array of processed (cropped and resized) images.
            - y (np.array): NumPy array of corresponding numerical labels.
            - label_to_numeric (dict): Mapping from string labels to numerical indices.
            - numeric_to_label (dict): Mapping from numerical indices to string labels.
    """
    # Check if the base_images_path exists
    if not os.path.exists(base_images_path):
        print(f"Error: The base directory '{base_images_path}' does not exist.")
        sys.exit(1) # Exit the script if the base path is invalid
    print(f"Base images path: {base_images_path}")

    # --- Dynamically collect labels from subdirectories ---
    # Get all subdirectory names within BASE_IMAGES_PATH
    actual_folder_labels = [name for name in os.listdir(base_images_path)
                            if os.path.isdir(os.path.join(base_images_path, name))]

    # Use all detected folder names as labels and sort them for consistent numerical mapping
    labels = sorted(actual_folder_labels)

    # Check if any labels were found
    if not labels:
        print("Error: No label directories found in the base path. Cannot proceed.")
        return np.array([]), np.array([]), {}, {}

    # Create a mapping from string label to numerical index
    label_to_numeric = {label: i for i, label in enumerate(labels)}
    numeric_to_label = {i: label for i, label in enumerate(labels)}
    num_classes = len(labels) # This will now reflect the actual number of folders

    print(f"Detected {num_classes} unique labels/classes dynamically from directories.")

    # Define a list of common image file extensions
    IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp')

    # Define the target size for resizing images
    TARGET_IMAGE_SIZE = (64, 64) # (width, height) - Consistent with detector for training

    # Lists to store our processed images and their corresponding numerical labels
    all_images = []
    all_numerical_labels = []

    print("\n--- Starting Image Preprocessing and Collection (with MediaPipe Hand Cropping) ---")

    # Initialize MediaPipe Hands for processing static training images
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True, # Set to True for processing static images from disk
        max_num_hands=2,         # Detect a single hand
        min_detection_confidence=0.6
    )

    # Iterate through each label
    for label in labels:
        # Construct the full path to the directory for the current label
        label_dir_path = os.path.join(base_images_path, label)

        # Check if the directory exists (should always exist now, as labels come from existing dirs)
        if not os.path.exists(label_dir_path):
            print(f"Warning: Directory '{label_dir_path}' not found. Skipping label '{label}'.")
            continue

        print(f"Processing images for label: {label} from {label_dir_path}")

        # --- Renaming Logic Start ---
        current_image_files = [f for f in os.listdir(label_dir_path)
                               if os.path.isfile(os.path.join(label_dir_path, f)) and f.lower().endswith(IMAGE_EXTENSIONS)]

        max_existing_number = 0
        files_to_rename = []
        desired_name_pattern = re.compile(rf"^{re.escape(label)}_(\d+)\.\w+$", re.IGNORECASE)

        for img_name in current_image_files:
            match = desired_name_pattern.match(img_name)
            if match:
                num = int(match.group(1))
                if num > max_existing_number:
                    max_existing_number = num
            else:
                files_to_rename.append(img_name)

        rename_counter = max_existing_number + 1

        if files_to_rename:
            print(f"  Renaming {len(files_to_rename)} files in '{label_dir_path}'...")
            for old_img_name in files_to_rename:
                file_base, file_ext = os.path.splitext(old_img_name)
                new_img_name = f"{label}_{rename_counter}{file_ext.lower()}" # Ensure extension is lowercase

                old_image_path = os.path.join(label_dir_path, old_img_name)
                new_image_path = os.path.join(label_dir_path, new_img_name)

                if not os.path.exists(new_image_path):
                    try:
                        os.rename(old_image_path, new_image_path)
                        print(f"    Renamed '{old_img_name}' to '{new_img_name}'")
                        rename_counter += 1
                    except OSError as e:
                        print(f"    Error renaming '{old_img_name}': {e}")
                else:
                    print(f"    Skipping rename for '{old_img_name}': '{new_img_name}' already exists.")
                    rename_counter += 1

        # --- Renaming Image Logic End ---

        # --- Image Reading, MediaPipe Cropping, Resizing Logic Start ---
        # Re-list files in case renaming happened
        processed_image_filenames = [f for f in os.listdir(label_dir_path)
                                     if os.path.isfile(os.path.join(label_dir_path, f)) and f.lower().endswith(IMAGE_EXTENSIONS)]

        for img_name in processed_image_filenames:
            image_path = os.path.join(label_dir_path, img_name)

            try:
                img = cv2.imread(image_path)

                if img is None:
                    print(f"Could not read image: {image_path}. Skipping.")
                    continue
                else:
                    # Convert BGR to RGB for MediaPipe processing
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    results = hands.process(img_rgb) # Process image with MediaPipe

                    # Only process if a hand is detected
                    if results.multi_hand_landmarks:
                        # Assuming only one hand per image for sign language
                        for hand_landmarks in results.multi_hand_landmarks:
                            # Get bounding box coordinates for the hand from landmarks
                            h, w, _ = img.shape # Use the dimensions of the original image
                            x_coords = [landmark.x for landmark in hand_landmarks.landmark]
                            y_coords = [landmark.y for landmark in hand_landmarks.landmark]

                            x_min = int(min(x_coords) * w)
                            y_min = int(min(y_coords) * h)
                            x_max = int(max(x_coords) * w)
                            y_max = int(max(y_coords) * h)

                            # Add padding to the bounding box to ensure full hand is captured
                            padding_px = 30 # Consistent with detector's padding
                            x_min = max(0, x_min - padding_px)
                            y_min = max(0, y_min - padding_px)
                            x_max = min(w, x_max + padding_px)
                            y_max = min(h, y_max + padding_px)

                            # Crop the hand region from the original image
                            cropped_hand = img[y_min:y_max, x_min:x_max]

                            # Check if crop resulted in an empty image (e.g., if bounding box invalid)
                            if cropped_hand.size == 0:
                                print(f"Warning: Cropped hand region is empty for {image_path}. Skipping.")
                                continue

                            # Resize the cropped hand to the target size
                            resized_img = cv2.resize(cropped_hand, TARGET_IMAGE_SIZE)
                            # The MobileNetV2 preprocessing layer in the trainer expects [0, 255] values.
                            processed_img_for_trainer = resized_img.astype(np.float32)

                            # Append the processed image and its numerical label
                            all_images.append(processed_img_for_trainer)
                            all_numerical_labels.append(label_to_numeric[label])
                    else:
                        print(f"Warning: No hand detected in image {image_path}. Skipping for training.")
                        continue # Skip images where no hand is detected by MediaPipe

            except Exception as e:
                print(f"Error processing image {image_path}: {e}")

    cv2.destroyAllWindows() # Ensure any leftover OpenCV windows are closed
    hands.close() # Release MediaPipe resources
    print("\n--- Image Preprocessing and Collection Complete ---")
    print(f"Collected {len(all_images)} images.")

    if not all_images:
        print("No images were collected. Returning empty data.")
        return np.array([]), np.array([]), {}, {}

    # Convert lists to NumPy arrays
    X = np.array(all_images)
    y = np.array(all_numerical_labels)

    # Ensure X has 3 channels if it's grayscale (H, W) or (H, W, 1)
    # MediaPipe operates on color (3 channels), so `img` from `cv2.imread` is usually (H, W, 3).
    # If it somehow becomes (H, W), we add a channel.
    if len(X.shape) == 3: # This means it's (num_samples, H, W)
        X = np.expand_dims(X, axis=-1) # Add channel dimension to make it (num_samples, H, W, 1)
    elif X.shape[-1] == 1: # If it's (num_samples, H, W, 1), convert to (num_samples, H, W, 3) for MobileNetV2
        X = np.repeat(X, 3, axis=-1)

    print(f"Final collected data shape: X={X.shape}, y={y.shape}")

    # --- THIS IS THE FINAL RETURN OF THE FUNCTION ---
    return X, y, label_to_numeric, numeric_to_label

if __name__ == "__main__":
    BASE_IMAGES_PATH = 'C:/Users/adede/Documents/FYPPython/Tensorflow/workspace/images/myimages'
    X_data, y_data, l_to_n, n_to_l = collect_and_process_images(BASE_IMAGES_PATH)

    print("\n--- Data collected from sign_language_data_processor.py (for testing) ---")
    print(f"Shape of X_data: {X_data.shape}")
    print(f"Shape of y_data: {y_data.shape}")
    print("Label Mapping:", n_to_l)
