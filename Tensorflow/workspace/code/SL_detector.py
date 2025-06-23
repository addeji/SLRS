import cv2
import numpy as np
import tensorflow as tf
import os
import json
import time
import mediapipe as mp  # Import MediaPipe
from tensorflow.keras.applications.mobilenet_v2 import \
    preprocess_input as mobilenet_v2_preprocess_input  # Import MobileNetV2 specific preprocessing


def run_sign_language_detector(model_path='sign_language_model_transfer.keras', label_map_path=None):
    """
    Loads a trained sign language recognition model (trained with MobileNetV2 transfer learning)
    and its corresponding label mapping to perform real-time detection using a webcam,
    employing MediaPipe Hands for hand detection and cropping, and applying MobileNetV2's
    required preprocessing, displaying predictions and accumulating a sentence.

    Args:
        model_path (str): The path to the saved Keras model file (e.g., 'sign_language_model_transfer.keras').
        label_map_path (str, optional): The path to the JSON file containing the label mapping.
                                        If None, it will try to derive it from model_path.
    """
    # --- 1. Load the Trained Model ---
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at '{model_path}'.")
        print("Please ensure you have trained the model and it's saved in the correct location and format (.keras).")
        return

    try:
        model = tf.keras.models.load_model(model_path)
        print(f"Model loaded successfully from '{model_path}'")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Ensure TensorFlow is installed and the model file is not corrupted or is in the correct format.")
        return

    # --- 2. Load Label Mapping Dynamically ---
    # NOTE: This part assumes you are either running `save_labels_debug.py`
    # or manually providing the label_map_path, as the trainer no longer saves it directly.
    if label_map_path is None:
        # Default label map path if not provided (assumes same base name as model + '_labels.json')
        label_map_path = 'C:/Users/adede/Documents/FYPPython/Tensorflow/workspace/code/SLR_model_labels_debug.json'

    numeric_to_label = {}
    if not os.path.exists(label_map_path):
        print(f"Error: Label mapping file not found at '{label_map_path}'.")
        print(
            "Please ensure the label mapping was saved during training (e.g., using save_labels_debug.py) and is accessible.")
        print(
            "Alternatively, copy the 'Numerical to Label' mapping from your trainer's output and hardcode it into this script.")
        return
    try:
        with open(label_map_path, 'r') as f:
            loaded_map = json.load(f)
            numeric_to_label = {int(k): v for k, v in loaded_map.items()}
        print(f"Label mapping loaded successfully from '{label_map_path}'")
    except Exception as e:
        print(f"Error loading label mapping: {e}")
        print("Ensure the JSON file is valid and accessible.")
        return

    # --- 3. Define Preprocessing Parameters (MUST match the training script) ---
    TARGET_IMAGE_SIZE = (64, 64)  # (width, height) - Consistent with sign_language_data_processor.py

    # --- 4. UI Layout and Drawing Parameters ---
    VIDEO_WIDTH = 640
    VIDEO_HEIGHT = 480
    TEXT_AREA_WIDTH = 400
    PADDING = 20

    WINDOW_WIDTH = VIDEO_WIDTH + TEXT_AREA_WIDTH + PADDING * 3
    WINDOW_HEIGHT = VIDEO_HEIGHT + PADDING * 2

    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE_PREDICTION = 1.0
    FONT_THICKNESS_PREDICTION = 2
    TEXT_COLOR_PREDICTION = (0, 255, 0)  # Green

    FONT_SCALE_SENTENCE = 0.8
    FONT_THICKNESS_SENTENCE = 1
    TEXT_COLOR_SENTENCE = (255, 255, 255)  # White
    BACKGROUND_COLOR = (30, 30, 30)  # Dark gray for the overall window and text box

    # --- 5. Detection Logic Parameters ---
    MIN_CONFIDENCE_THRESHOLD = 0.7
    CONFIRMATION_COOLDOWN_SECONDS = 1.5

    # --- 6. Initialize MediaPipe Hands ---
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,  # Process video stream
        max_num_hands=2,  # Detect a single hand
        min_detection_confidence=0.7,  # Higher confidence for initial detection
        min_tracking_confidence=0.6  # Confidence to continue tracking
    )
    mp_drawing = mp.solutions.drawing_utils  # For drawing landmarks

    # --- 7. Initialize Webcam Capture ---
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video stream (webcam).")
        print("Please check if your webcam is connected and not in use by another application.")
        return

    print("\n--- Starting Real-time Sign Language Detector ---")
    print(f"Total {len(numeric_to_label)} classes detected.")
    print("Instructions:")
    print("  - Position your hand clearly in front of the camera.")
    print("  - Press 'Spacebar' to confirm the currently predicted sign and add it to the sentence.")
    print("  - Press 'c' to clear the current sentence.")
    print("  - Press 'q' to quit the detector window.")

    current_sentence = []
    last_confirmed_word_time = time.time() - CONFIRMATION_COOLDOWN_SECONDS
    confirmed_message_display_until = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Failed to grab frame, exiting...")
            break

        frame = cv2.flip(frame, 1)  # Mirror the frame for intuitive use

        # Convert the BGR image to RGB for MediaPipe
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the image with MediaPipe Hands
        results = hands.process(image_rgb)

        predicted_label = "No Hand Detected"  # Default
        confidence = 0.0

        # Create a blank canvas for the combined display
        display_canvas = np.full((WINDOW_HEIGHT, WINDOW_WIDTH, 3), BACKGROUND_COLOR, dtype=np.uint8)

        # --- Hand Detection and Prediction Logic ---
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks on the frame for visualization
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Get bounding box coordinates for the hand
                h, w, c = frame.shape
                x_coords = [landmark.x for landmark in hand_landmarks.landmark]
                y_coords = [landmark.y for landmark in hand_landmarks.landmark]

                x_min = int(min(x_coords) * w)
                y_min = int(min(y_coords) * h)
                x_max = int(max(x_coords) * w)
                y_max = int(max(y_coords) * h)

                # Add some padding to the bounding box
                padding_px = 30
                x_min = max(0, x_min - padding_px)
                y_min = max(0, y_min - padding_px)
                x_max = min(w, x_max + padding_px)
                y_max = min(h, y_max + padding_px)

                # Draw bounding box on the frame
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)  # Blue rectangle

                # Crop the hand region
                cropped_hand = frame[y_min:y_max, x_min:x_max]

                if cropped_hand.size == 0:
                    predicted_label = "Invalid Crop"
                    confidence = 0.0
                else:
                    # Preprocess the cropped hand for model prediction:
                    # 1. Resize to target size
                    input_frame_for_model = cv2.resize(cropped_hand, TARGET_IMAGE_SIZE)
                    # 2. Convert to float32 (pixels still in [0, 255] range)
                    input_frame_for_model = input_frame_for_model.astype(np.float32)
                    # 3. Apply MobileNetV2's specific preprocessing (scales to [-1, 1])
                    input_frame_for_model = mobilenet_v2_preprocess_input(input_frame_for_model)
                    # 4. Add batch dimension
                    input_frame_for_model = np.expand_dims(input_frame_for_model, axis=0)

                    # Make Prediction
                    predictions = model.predict(input_frame_for_model, verbose=0)
                    predicted_class_index = np.argmax(predictions)
                    confidence = predictions[0][predicted_class_index]
                    predicted_label = numeric_to_label.get(predicted_class_index, "Unknown")
        else:
            predicted_label = "No Hand Detected"
            confidence = 0.0

        # --- Display Live Prediction on Video Feed ---
        live_prediction_text = f"Sign: {predicted_label} ({confidence * 100:.2f}%)"

        (text_width, text_height), baseline = cv2.getTextSize(live_prediction_text, FONT, FONT_SCALE_PREDICTION,
                                                              FONT_THICKNESS_PREDICTION)

        # Draw background for prediction text on the original frame
        cv2.rectangle(frame, (PADDING, PADDING), (PADDING + text_width + PADDING, PADDING + text_height + PADDING),
                      BACKGROUND_COLOR, -1)
        cv2.putText(frame, live_prediction_text, (PADDING + 10, PADDING + text_height + 5), FONT, FONT_SCALE_PREDICTION,
                    TEXT_COLOR_PREDICTION, FONT_THICKNESS_PREDICTION, cv2.LINE_AA)

        # Copy the processed video frame to the left side of the display_canvas
        resized_video_frame = cv2.resize(frame, (VIDEO_WIDTH, VIDEO_HEIGHT))
        display_canvas[PADDING: PADDING + VIDEO_HEIGHT, PADDING: PADDING + VIDEO_WIDTH] = resized_video_frame

        # --- Draw Dialogue Box (Right Side) ---
        dialogue_box_x_start = PADDING * 2 + VIDEO_WIDTH
        dialogue_box_y_start = PADDING
        dialogue_box_x_end = WINDOW_WIDTH - PADDING
        dialogue_box_y_end = WINDOW_HEIGHT - PADDING

        cv2.rectangle(display_canvas,
                      (dialogue_box_x_start, dialogue_box_y_start),
                      (dialogue_box_x_end, dialogue_box_y_end),
                      (60, 60, 60),
                      -1)

        title_text = "Generated Sentence:"
        (title_w, title_h), _ = cv2.getTextSize(title_text, FONT, FONT_SCALE_SENTENCE * 1.2,
                                                FONT_THICKNESS_SENTENCE + 1)
        cv2.putText(display_canvas, title_text,
                    (dialogue_box_x_start + PADDING, dialogue_box_y_start + PADDING + title_h),
                    FONT, FONT_SCALE_SENTENCE * 1.2, (255, 255, 0), FONT_THICKNESS_SENTENCE + 1, cv2.LINE_AA)

        cv2.line(display_canvas,
                 (dialogue_box_x_start + PADDING, dialogue_box_y_start + PADDING + title_h + 10),
                 (dialogue_box_x_end - PADDING, dialogue_box_y_start + PADDING + title_h + 10),
                 (100, 100, 100), 1)

        sentence_str = " ".join(current_sentence)
        max_text_width = TEXT_AREA_WIDTH - PADDING * 2

        wrapped_sentence = []
        line = ""
        words = sentence_str.split(' ')
        for word in words:
            test_line = line + " " + word if line else word
            (test_w, _), _ = cv2.getTextSize(test_line, FONT, FONT_SCALE_SENTENCE, FONT_THICKNESS_SENTENCE)
            if test_w > max_text_width:
                wrapped_sentence.append(line)
                line = word
            else:
                line = test_line
        wrapped_sentence.append(line)

        y_offset = dialogue_box_y_start + PADDING * 3 + title_h
        for line_text in wrapped_sentence:
            cv2.putText(display_canvas, line_text,
                        (dialogue_box_x_start + PADDING, y_offset),
                        FONT, FONT_SCALE_SENTENCE, TEXT_COLOR_SENTENCE, FONT_THICKNESS_SENTENCE, cv2.LINE_AA)
            y_offset += (cv2.getTextSize(line_text, FONT, FONT_SCALE_SENTENCE, FONT_THICKNESS_SENTENCE)[0][1] + 15)

        if time.time() < confirmed_message_display_until:
            msg = "Word Added!"
            (msg_w, msg_h), _ = cv2.getTextSize(msg, FONT, FONT_SCALE_PREDICTION * 0.8, FONT_THICKNESS_PREDICTION)
            msg_x = PADDING + VIDEO_WIDTH // 2 - msg_w // 2
            msg_y = PADDING + VIDEO_HEIGHT - PADDING - msg_h
            cv2.putText(display_canvas, msg, (msg_x, msg_y), FONT, FONT_SCALE_PREDICTION * 0.8, (0, 255, 255),
                        FONT_THICKNESS_PREDICTION, cv2.LINE_AA)

        cv2.imshow('Sign Language Detector', display_canvas)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("Quitting detector.")
            break
        elif key == ord(' '):
            if predicted_label != "No Hand Detected" and predicted_label != "Invalid Crop" and confidence > MIN_CONFIDENCE_THRESHOLD and time.time() - last_confirmed_word_time >= CONFIRMATION_COOLDOWN_SECONDS:
                current_sentence.append(predicted_label)
                print(f"Added word: {predicted_label}. Current sentence: {' '.join(current_sentence)}")
                last_confirmed_word_time = time.time()
                confirmed_message_display_until = time.time() + 1.5
        elif key == ord('c'):
            current_sentence = []
            print("Sentence cleared.")

    cap.release()
    hands.close()  # Release MediaPipe resources
    cv2.destroyAllWindows()


if __name__ == "__main__":
    MODEL_PATH = 'sign_language_model_transfer.keras'
    run_sign_language_detector(MODEL_PATH)
