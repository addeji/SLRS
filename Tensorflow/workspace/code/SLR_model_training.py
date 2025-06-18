import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import os
from tensorflow.keras.callbacks import ModelCheckpoint # Imports ModelCheckpoint lib

# Imports the data collection and processing function
from SLR_data_image_processor import collect_and_process_images

def train_sign_language_model(base_images_path, model_save_name='sign_language_model.keras', epochs=100):
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    """
    collects processed image data and trains a CNN model for sign language recognition.
    the batch size is dynamically determined based on the number of images.
    includes model checkpointing to save the best performing model during training.

    Args:
        base_images_path (str): The base directory where image subfolders are located.
        model_save_name (str): The filename for saving the final trained model.
        epochs (int): Number of training epochs.
    """
    # 1. Collect and process images using the function from the other file
    X, y, label_to_numeric, numeric_to_label = collect_and_process_images(base_images_path)

    if X.size == 0: # Check if any images were collected
        print("No data collected for training. Exiting model training.")
        return

    num_classes = len(label_to_numeric)
    input_shape = X.shape[1:] # Get (height, width, channels) from collected data
    total_images = len(X) # Get the total number of collected images

    print("\n--- Starting Model Training Pipeline ---")

    print(f"Input image shape for model: {input_shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Number of classes: {num_classes}")
    print(f"Total collected images: {total_images}")

    # ---block to dynamically Determine Batch Size ---
    # Define a helper function to find the largest power of 2 less than or equal to n
    def floor_power_of_2(n):
        if n <= 0: return 1 # Handle edge cases
        p = 1
        while p * 2 <= n:
            p *= 2
        return p

    MAX_BATCH_SIZE_CAP = 128 # Maximum practical batch size (adjust based on your GPU memory)
    MIN_BATCH_SIZE = 16    # Minimum viable batch size for stability

    dynamic_batch_size = MIN_BATCH_SIZE # Starts with a minimum

    if total_images >= MIN_BATCH_SIZE:
        # We'll aim for a batch size that is a power of 2 and roughly 5-10% of the total dataset,
        # but capped by MAX_BATCH_SIZE_CAP and at least MIN_BATCH_SIZE.
        # We'll consider batch sizes from 16, 32, 64, 128.
        if total_images >= 128:
            dynamic_batch_size = min(MAX_BATCH_SIZE_CAP, floor_power_of_2(total_images // 8)) # Aim for 1/8th of the total
        elif total_images >= 64:
            dynamic_batch_size = 64
        elif total_images >= 32:
            dynamic_batch_size = 32
        else: # For very small datasets where total_images < 32
            dynamic_batch_size = MIN_BATCH_SIZE

    # Ensure the dynamic_batch_size is within the practical range and is a power of 2
    # This final adjustment ensures it's one of the preferred values [16, 32, 64, 128]
    if dynamic_batch_size > MAX_BATCH_SIZE_CAP:
        dynamic_batch_size = MAX_BATCH_SIZE_CAP
    if dynamic_batch_size < MIN_BATCH_SIZE:
        dynamic_batch_size = MIN_BATCH_SIZE

    # If the calculated batch size is not a power of 2 (due to min/max caps), force it to the nearest power of 2 below it
    dynamic_batch_size = floor_power_of_2(dynamic_batch_size)


    print(f"Dynamically determined batch size: {dynamic_batch_size}")
    batch_size = dynamic_batch_size # Use this variable for training

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"Training images: {X_train.shape}, Training labels: {y_train.shape}")
    print(f"Test images: {X_test.shape}, Test labels: {y_test.shape}")

    # Build the CNN Model
    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=input_shape),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),

        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])

    # Displays model summary
    model.summary()

    # --- Setup Model Checkpointing ---
    # Directory where checkpoints will be saved
    checkpoint_dir = 'C:/Users/adede/Documents/FYPPython/Tensorflow/workspace/checkpoint'
    os.makedirs(checkpoint_dir, exist_ok=True) # Creates the directory if it doesn't exist

    # Define the checkpoint file path format
    # This will save files like 'checkpoint_epoch_05_val_accuracy_0.92.h5'
    checkpoint_filepath = os.path.join(checkpoint_dir, 'checkpoint_epoch_{epoch:02d}_val_accuracy_{val_accuracy:.4f}.h5')

    # Create the ModelCheckpoint callback
    # monitor='val_accuracy': save based on validation accuracy
    # save_best_only=True: only save the model when validation accuracy improves
    # mode ='max': because we want to maximize validation accuracy
    # verbose=1: to see messages when a model is saved
    model_checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )

    # Train the model
    print(f"\nTraining the model for {epochs} epochs with batch size {batch_size}...")
    history = model.fit(X_train, y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_split=0.1,
                        callbacks=[model_checkpoint_callback]) # Add the checkpoint callback here

    # Evaluate the model on the test set
    print("\nEvaluating the model on the test set...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=2)
    print(f"\nTest accuracy: {test_accuracy*100:.2f}%")

    # Saves the final trained model (optional, as best models are saved by checkpoint)
    try:
        model.save(model_save_name)
        print(f"Final model saved successfully to {model_save_name}")
    except Exception as e:
        print(f"Error saving final model: {e}")

    print("\n--- Model Training Pipeline Complete ---")
    print(f"The best performing models are saved in the '{checkpoint_dir}' directory.")
    print(f"The final model (last epoch) is saved at: {os.path.abspath(model_save_name)}")
    print("\n--- Label Mapping ---")
    print("Numerical to Label:")
    for num_idx, char_label in numeric_to_label.items():
        print(f"  {num_idx}: '{char_label}'")
    print("This mapping is crucial when using your model for predictions.")

# This ensures the training process runs when the file is executed directly
if __name__ == "__main__":
    # IMPORTANT: Adjust this path to the absolute path where your 'collected images' folder is located
    TRAINING_DATA_BASE_PATH = 'C:/Users/adede/Documents/FYPPython/Tensorflow/workspace/images/collectedimages'
    train_sign_language_model(TRAINING_DATA_BASE_PATH, epochs=100)
