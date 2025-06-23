import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import os
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2 # Import MobileNetV2
from tensorflow.keras.layers import Dense, Flatten, Dropout, Input
from tensorflow.keras import Model # Import Model for functional API

# Imports the data collection and processing function
from SLR_data_image_processor import collect_and_process_images
from plot_learning_curve import plot_learning_curves

def train_sign_language_model(base_images_path, model_save_name='sign_language_model_transfer.keras', epochs=150):
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

    """
    Collects processed image data and trains a CNN model for sign language recognition
    using MobileNetV2 for transfer learning.
    The batch size is dynamically determined based on the number of images.
    Includes model checkpointing to save the best performing model during training.
    Includes data augmentation for improved robustness.
    NOTE: NUMERICAL-TO-LABEL MAPPING IS NOT SAVED TO A JSON FILE IN THIS SCRIPT.

    Args:
        base_images_path (str): The base directory where image subfolders are located.
        model_save_name (str): The filename for saving the final trained model.
        epochs (int): Number of training epochs for the new head.
    """
    # 1. Collect and process images (with MediaPipe hand cropping)
    X, y, label_to_numeric, numeric_to_label = collect_and_process_images(base_images_path)

    if X.size == 0:
        print("No data collected for training. Exiting model training.")
        return

    num_classes = len(label_to_numeric)
    # The input_shape for the model should match the TARGET_IMAGE_SIZE from data_processor
    # and include the channel dimension (e.g., (64, 64, 3))
    input_shape = X.shape[1:]
    total_images = len(X)

    print("\n--- Starting Model Training Pipeline with Transfer Learning (MobileNetV2) ---")

    print(f"Input image shape for model: {input_shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Number of classes: {num_classes}")
    print(f"Total collected images: {total_images}")

    # --- Dynamic Batch Size Determination ---
    def floor_power_of_2(n):
        if n <= 0: return 1
        p = 1
        while p * 2 <= n:
            p *= 2
        return p

    MAX_BATCH_SIZE_CAP = 128
    MIN_BATCH_SIZE = 16
    dynamic_batch_size = MIN_BATCH_SIZE

    if total_images >= MIN_BATCH_SIZE:
        if total_images >= 128:
            dynamic_batch_size = min(MAX_BATCH_SIZE_CAP, floor_power_of_2(total_images // 8))
        elif total_images >= 64:
            dynamic_batch_size = 64
        elif total_images >= 32:
            dynamic_batch_size = 32
        else:
            dynamic_batch_size = MIN_BATCH_SIZE

    dynamic_batch_size = floor_power_of_2(dynamic_batch_size)
    print(f"Dynamically determined batch size: {dynamic_batch_size}")
    batch_size = dynamic_batch_size

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"Training images: {X_train.shape}, Training labels: {y_train.shape}")
    print(f"Test images: {X_test.shape}, Test labels: {y_test.shape}")

    # --- Setup Data Augmentation ---
    datagen = ImageDataGenerator(
        rotation_range=0.01,
        width_shift_range=0.01,
        height_shift_range=0.01,
        zoom_range=0,
        horizontal_flip=False,
        brightness_range=[0.9, 1.1],
        fill_mode='nearest'
    )
    train_generator = datagen.flow(X_train, y_train, batch_size=batch_size, shuffle=True)

    # --- Build the Transfer Learning Model ---
    # 1. Load the MobileNetV2 base model
    # include_top=False: Excludes the ImageNet classification head
    # weights='imagenet': Uses pre-trained weights from ImageNet
    # input_shape: Should match the size of your processed images (64, 64, 3)
    base_model = MobileNetV2(input_shape=input_shape,
                             include_top=False,
                             weights='imagenet')

    # 2. Freeze the base model layers
    # This prevents their weights from being updated during the first training phase
    base_model.trainable = False

    # 3. Create the custom classification head
    # Use the Functional API for more flexibility
    inputs = Input(shape=input_shape)
    # The pre-trained models often expect inputs scaled in a specific way,
    # MobileNetV2 expects values in [-1, 1]. Your current data is [0, 1].
    # Let's add a rescaling layer that MobileNetV2 normally uses.
    # Note: tf.keras.applications.mobilenet_v2.preprocess_input could be used directly,
    # but integrating it as a layer handles it transparently.
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs) # Apply MobileNetV2 specific preprocessing
    x = base_model(x, training=False) # Pass through the base model (training=False ensures batch norm layers are fixed)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)

    # Compile the model
    # Use a low learning rate when transfer learning to avoid corrupting pre-trained weights
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001), # Slightly lower default LR for new head
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])

    # Displays model summary
    model.summary()

    # --- Setup Model Checkpointing ---
    checkpoint_dir = 'C:/Users/adede/Documents/FYPPython/Tensorflow/workspace/checkpoint_transfer_learning'
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_filepath = os.path.join(checkpoint_dir, 'checkpoint_epoch_{epoch:02d}_val_accuracy_{val_accuracy:.4f}.keras')

    model_checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )

    # Train the new head of the model
    print(f"\nTraining the new classification head for {epochs} epochs with batch size {batch_size} (with Data Augmentation)...")
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=(X_test, y_test),
        callbacks=[model_checkpoint_callback],
        steps_per_epoch=len(X_train) // batch_size
    )

    # Evaluate the model on the test set
    print("\nEvaluating the model on the test set...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=2)
    print(f"\nTest accuracy: {test_accuracy*100:.2f}%")

    # Saves the final trained model
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

    # --- Call the plotting functions ---
    print("\n--- Generating Performance Visualizations ---")
    plot_learning_curves(history)

# This ensures the training process runs when the file is executed directly
if __name__ == "__main__":
    TRAINING_DATA_BASE_PATH = 'C:/Users/adede/Documents/FYPPython/Tensorflow/workspace/images/myimages'
    # Increase epochs for transfer learning (e.g., 50 for head, then more for fine-tuning)
    # We are setting a higher default here to allow enough training for the new head.
    train_sign_language_model(TRAINING_DATA_BASE_PATH, epochs=150) # Start with 50 epochs for head
