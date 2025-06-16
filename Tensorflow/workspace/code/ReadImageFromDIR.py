import os
import cv2 #  cv2 for reading and processing images
import uuid # to generate unique IDs for processed images
import string # Import the string module to easily get alphabet characters

# Defines the base directory where images are located
# This is the parent directory that contains subfolders for each label
BASE_IMAGES_PATH = 'C:/Users/adede/Documents/FYPPython/Tensorflow/workspace/images/collectedimages'


# You can easily add more custom words to this list
custom_labels = ['accept', 'go', 'stop', 'hello', 'thanks', 'yes', 'no', 'iloveyou']
labels = list(string.ascii_uppercase) + custom_labels

# line to only read images from the directories corresponding to these labels
# Only files with these extensions will be processed as images.
IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp')

# Iterate through each label
for label in labels:
    # Construct the full path to the directory for the current label
    label_dir_path = os.path.join(BASE_IMAGES_PATH, label)

    # Check if the directory exists
    if not os.path.exists(label_dir_path):
        print(f"Warning: Directory '{label_dir_path}' not found. Skipping label '{label}'.")
        continue

    print(f"Reading images for label: {label} from {label_dir_path}")

    # List all files in the current label's directory
    image_filenames = os.listdir(label_dir_path)

    for img_name in image_filenames:
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

                # --- You can add your image processing/training code here ---
                # For demonstration, we'll just show the original color image
                cv2.imshow(f'Image for {label}', img)

                # Waits for a key press to show the next image (optional)
                # Press 'q' to quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break # Exit inner loop
            except Exception as e:
                print(f"Error processing image {image_path}: {e}")
        # If it's not a file or not an image (based on extension), we simply skip it.
    # If 'q' was pressed, break from the outer loop as well
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
print("Finished reading images.")
