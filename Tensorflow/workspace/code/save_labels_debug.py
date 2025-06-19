import os
import json
import sys
import numpy as np  # Needed if collect_and_process_images returns np arrays or uses np internally

# Import the data collection and processing function from your existing file
from SLR_data_image_processor import collect_and_process_images


def debug_save_labels():
    """
    Collects labels using SLR_data_image_processor and attempts to save them
    to a JSON file at a specified absolute path.
    """
    # This path points to the 'collectedimages' folder.
    BASE_IMAGES_PATH = 'C:/Users/adede/Documents/FYPPython/Tensorflow/workspace/images/collectedimages'


    FIXED_SAVE_DIRECTORY = 'C:/Users/adede/Documents/FYPPython/Tensorflow/workspace/code'
    LABEL_FILENAME = 'SLR_model_labels_debug.json'  # A distinct name for debugging

    print(f"\n--- Starting Debugging Script: save_labels_debug.py ---")
    print(f"Base images path for data collection: {BASE_IMAGES_PATH}")
    print(f"Target directory for saving label JSON: {FIXED_SAVE_DIRECTORY}")

    # 1. Attempts to create the save directory
    print(f"\nAttempting to create save directory: {FIXED_SAVE_DIRECTORY}")
    try:
        os.makedirs(FIXED_SAVE_DIRECTORY, exist_ok=True)
        print(
            f"Directory creation status: {'Created' if os.path.exists(FIXED_SAVE_DIRECTORY) else 'Already exists or created'}")
    except Exception as e:
        print(f"Error creating directory {FIXED_SAVE_DIRECTORY}: {e}")
        print("Please check permissions for the target directory.")
        sys.exit(1)  # Exit if directory cannot be created

    # 2. Collect data and get label mapping
    print("\nAttempting to collect and process image labels...")
    X, y, label_to_numeric, numeric_to_label = collect_and_process_images(BASE_IMAGES_PATH)

    if not numeric_to_label:
        print("Error: No labels were collected from the image data processor. Cannot save label map.")
        return

    # Print the collected labels to ensure they are correct
    print(f"\nCollected {len(numeric_to_label)} labels:")
    for num_idx, char_label in numeric_to_label.items():
        print(f"  {num_idx}: '{char_label}'")

    # 3. Construct the full path for the JSON file
    label_map_path = os.path.join(FIXED_SAVE_DIRECTORY, LABEL_FILENAME)
    print(f"\nAttempting to save label mapping to: {label_map_path}")

    # 4. Attempt to save the label mapping to JSON
    try:
        # Convert int keys to string for JSON serialization (JSON keys must be strings)
        numeric_to_label_str_keys = {str(k): v for k, v in numeric_to_label.items()}
        with open(label_map_path, 'w') as f:
            json.dump(numeric_to_label_str_keys, f, indent=4)
        print(f"Label mapping saved successfully to {label_map_path}")
        print(f"Please check this exact path to verify the file: {os.path.abspath(label_map_path)}")
    except Exception as e:
        print(f"ERROR: Failed to save label mapping: {e}")
        print("This could be due to permissions, an invalid path, or issues with the dictionary content.")
        print("Please double-check FIXED_SAVE_DIRECTORY and ensure you have write access.")

    print("\n--- Debugging Script Finished ---")


if __name__ == "__main__":
    debug_save_labels()
