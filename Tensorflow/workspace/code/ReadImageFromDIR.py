import os
import re
import string

import cv2  # cv2 for reading and processing images
import numpy as np

# Defines the base directory where images are located
# This is the parent directory that contains subfolders for each label
BASE_IMAGES_PATH = 'C:/Users/adede/Documents/FYPPython/Tensorflow/workspace/images/collectedimages'


# You can easily add more custom words to this list
custom_labels = ['accept', 'go', 'stop', 'hello', 'why_while', 'yes', 'welcome', 'iloveyou']
labels = list(string.ascii_uppercase) + custom_labels

# line to only read images from the directories corresponding to these labels
# Only files with these extensions will be processed as images.
IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp')

# Define the target size for resizing images
TARGET_IMAGE_SIZE = (224, 224) # (width, height)

# Iterate through each label
for label in labels:
    # Construct the full path to the directory for the current label
    label_dir_path = os.path.join(BASE_IMAGES_PATH, label)

    # Check if the directory exists
    if not os.path.exists(label_dir_path):
        print(f"Warning: Directory '{label_dir_path}' not found. Skipping label '{label}'.")
        continue

    print(f"Reading images for label: {label} from {label_dir_path}")

    # --- Renaming Logic Start ---
    # first we'll go through and rename files that don't match the desired pattern.

    # Gets all current image files in the directory
    current_image_files = [f for f in os.listdir(label_dir_path)
                           if os.path.isfile(os.path.join(label_dir_path, f)) and f.lower().endswith(IMAGE_EXTENSIONS)]

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
                                 if os.path.isfile(os.path.join(label_dir_path, f)) and f.lower().endswith(IMAGE_EXTENSIONS)]

    for img_name in processed_image_filenames:
        # Constructs the full path to the image file
        image_path = os.path.join(label_dir_path, img_name)

        # Checks if it's a file AND if its extension is in our list of IMAGE_EXTENSIONS
        if os.path.isfile(image_path) and img_name.lower().endswith(IMAGE_EXTENSIONS):
            try:
                # Reads the image using OpenCV
                img = cv2.imread(image_path)

                # Checks if the image was loaded successfully
                if img is None:
                    print(f"Could not read image: {image_path}. Skipping.")
                    continue
                else:

                    # 1. Resizes the image
                    resized_img = cv2.resize(img, TARGET_IMAGE_SIZE)

                    # 2. Normalize the pixel values from 0-255 to 0.0-1.0
                    # Convert image to float32 type before division for precise normalization
                    normalized_img = resized_img.astype(np.float32) / 255.0


                # displays the image in a window
                    # So, we'll multiply by 255 to display, but for actual model input, use normalized_img
                    display_img = (normalized_img * 255).astype(np.uint8)
                    cv2.imshow(f'Processed Image for {label}: {img_name}', display_img)

                # Waits for a key input to close the image window
                # Press 'q' to quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break # Exit inner loop
            except Exception as e:
                print(f"Error processing image {image_path}: {e}")
        # If it's not a file or not an image (based on extension), we skip it.
    # If 'q' was pressed, break from the outer loop as well
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
print("Finished reading,renaming, resizing, and normalizing images.")
