import os
import cv2
import string
import re
import numpy as np
import sys # Import sys for potential graceful exit if re-enabling imshow

def collect_and_process_images(base_images_path):
    """
    Performs image preprocessing and data collection.

    Args:
        base_images_path (str): The base directory where image subfolders are located.

    Returns:
        tuple: A tuple containing:
            - X (np.array): NumPy array of processed images.
            - y (np.array): NumPy array of corresponding numerical labels.
            - label_to_numeric (dict): Mapping from string labels to numerical indices.
            - numeric_to_label (dict): Mapping from numerical indices to string labels.
    """
    # Check if the base_images_path existed
    # Defines the base directory where images are located
    # This is the parent directory that contains subfolders for each label
    if not os.path.exists(base_images_path):
        print(f"Error: The base directory '{base_images_path}' does not exist.")
        sys.exit(1) # Exit the script if the base path is invalid
    print(f"Base images path: {base_images_path}")

    # --- FIX RE-APPLIED: Dynamically collect labels from subdirectories ---
    # Get all subdirectory names within BASE_IMAGES_PATH
    actual_folder_labels = [name for name in os.listdir(base_images_path)
                            if os.path.isdir(os.path.join(base_images_path, name))]

    # Use all detected folder names as labels and sort them for consistent numerical mapping
    labels = sorted(actual_folder_labels)

    # Check if there are any labels after collection
    if not labels:
        print("Error: No label directories found in the base path. Cannot proceed.")
        return np.array([]), np.array([]), {}, {}

    # Create a mapping from string label to numerical index
    # This is crucial for training a classification model
    label_to_numeric = {label: i for i, label in enumerate(labels)}
    numeric_to_label = {i: label for i, label in enumerate(labels)}
    num_classes = len(labels) # This will now reflect the actual number of folders

    print(f"Detected {num_classes} unique labels/classes dynamically from directories.")


    # line to only read images from the directories corresponding to these labels
    # Only files with these extensions will be processed as images.
    image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp')

    # Define the target size for resizing images
    target_image_size = (64, 64) # (width, height) - Keeping it at 64x64 for memory efficiency


    # Lists to store our processed images and their corresponding numerical labels
    all_images = []
    all_numerical_labels = []

    print("\n--- Starting Image Preprocessing and Collection ---")

    # Iterate through each label
    for label in labels:
        # Construct the full path to the directory for the current label
        label_dir_path = os.path.join(base_images_path, label)

        # Check if the directory exists (should always exist now, as labels come from existing dirs)
        if not os.path.exists(label_dir_path):
            print(f"Warning: Directory '{label_dir_path}' not found. Skipping label '{label}'.")
            continue

        print(f"Reading images for label: {label} from {label_dir_path}")

        # --- Renaming Logic Start ---
        # first we'll go through and rename files that don't match the desired pattern.

        # Gets all current image files in the directory
        current_image_files = [f for f in os.listdir(label_dir_path)
                               if os.path.isfile(os.path.join(label_dir_path, f)) and f.lower().endswith(image_extensions)]

        max_existing_number = 0
        files_to_rename = []

        # Regex to match names like "A_1.png" or "accept_5.jpg"
        # Group 1 captures the label, Group 2 captures the number
        desired_name_pattern = re.compile(rf"^{re.escape(label)}_(\d+)\.\w+$", re.IGNORECASE)

        for img_name in current_image_files:
            match = desired_name_pattern.match(img_name)
            if match:
                # If it matches the desired pattern, update max_existing_number
                num = int(match.group(1))
                if num > max_existing_number:
                    max_existing_number = num
            else:
                # If it doesn't match, add it to the list for renaming
                files_to_rename.append(img_name)

        # Start numbering new files from max_existing_number + 1
        rename_counter = max_existing_number + 1

        # Perform the renaming for files that need it
        if files_to_rename:
            print(f"  Renaming {len(files_to_rename)} files in '{label_dir_path}'...")
            for old_img_name in files_to_rename:
                file_base, file_ext = os.path.splitext(old_img_name)
                new_img_name = f"{label}_{rename_counter}{file_ext.lower()}" # Ensure extension is lowercase

                old_image_path = os.path.join(label_dir_path, old_img_name)
                new_image_path = os.path.join(label_dir_path, new_img_name)

                # Check if the new name already exists (highly unlikely with sequential counter)
                if not os.path.exists(new_image_path):
                    try:
                        os.rename(old_image_path, new_image_path)
                        print(f"    Renamed '{old_img_name}' to '{new_img_name}'")
                        rename_counter += 1
                    except OSError as e:
                        print(f"    Error renaming '{old_img_name}': {e}")
                else:
                    print(f"    Skipping rename for '{old_img_name}': '{new_img_name}' already exists.")
                    rename_counter += 1 # Still increment to avoid future clashes

        # --- Renaming Image Logic End ---

        # --- Image Reading, Resizing, and Normalization Logic Start ---
        # Now, read all images (including those just renamed)
        # Re-list files in case renaming happened
        processed_image_filenames = [f for f in os.listdir(label_dir_path)
                                     if os.path.isfile(os.path.join(label_dir_path, f)) and f.lower().endswith(image_extensions)]

        for img_name in processed_image_filenames:
            # Constructs the full path to the image file
            image_path = os.path.join(label_dir_path, img_name)

            # Checks if it's a file AND if its extension is in our list of IMAGE_EXTENSIONS
            if os.path.isfile(image_path) and img_name.lower().endswith(image_extensions):
                try:
                    # Reads the image using OpenCV
                    img = cv2.imread(image_path)

                    # Checks if the image was loaded successfully
                    if img is None:
                        print(f"Could not read image: {image_path}. Skipping.")
                        continue
                    else:
                        # 1. Resizes the image
                        resized_img = cv2.resize(img, target_image_size)
                        # 2. Normalize the pixel values from 0-255 to 0.0-1.0
                        # Convert image to float32 type before division for precise normalization
                        normalized_img = resized_img.astype(np.float32) / 255.0

                        #Appends the processed image and its corresponding numerical label to the lists
                        all_images.append(normalized_img)
                        all_numerical_labels.append(label_to_numeric[label])

                except Exception as e:
                    print(f"Error processing image {image_path}: {e}")
            # If it's not a file or not an image (based on extension), we skip it.

    cv2.destroyAllWindows() # Ensure any leftover windows are closed
    print("\n--- Image Preprocessing and Collection Complete ---")
    print(f"Collected {len(all_images)} images.")

    if not all_images:
        print("No images were collected. Returning empty data.")
        # Return empty arrays/dicts if no images are collected
        return np.array([]), np.array([]), {}, {}

    # Convert lists to NumPy arrays
    X = np.array(all_images)
    y = np.array(all_numerical_labels)

    # Reshape X for CNN input (add channel dimension if missing, assuming color images here)
    if len(X.shape) == 3: # If shape is (num_samples, height, width) (meaning it's grayscale without channel dim)
        # Check if the last dimension is missing or is not 3 (color image) or 1 (grayscale).
        # This prevents adding an extra dimension if it's already (H, W, C) where C is 3.
        # cv2.imread usually returns (H, W, 3) for color or (H, W) for grayscale.
        # If it returns (H, W), we need to expand to (H, W, 1).
        if X.shape[-1] != 3: # If the last dimension is not 3, assume it's grayscale (H, W) and expand
            X = np.expand_dims(X, axis=-1) # Corrected assignment to 'X'

    # This print statement is now correctly placed after X and y are guaranteed to be assigned
    print(f"Final collected data shape: X={X.shape}, y={y.shape}")

    # --- THIS IS THE FINAL RETURN OF THE FUNCTION ---
    return X, y, label_to_numeric, numeric_to_label

# This block allows you to test this file independently, if needed
if __name__ == "__main__":
    # IMPORTANT: Adjust this path to the absolute path on your system for testing
    BASE_IMAGES_PATH = 'C:/Users/adede/Documents/FYPPython/Tensorflow/workspace/images/collectedimages'
    # Or for a more robust setup, you might pass it as an argument or from a config

    X_data, y_data, l_to_n, n_to_l = collect_and_process_images(BASE_IMAGES_PATH)

    print("\n--- Data collected from sign_language_data_processor.py (for testing) ---")
    print(f"Shape of X_data: {X_data.shape}")
    print(f"Shape of y_data: {y_data.shape}")
    print("Label Mapping:", n_to_l)
