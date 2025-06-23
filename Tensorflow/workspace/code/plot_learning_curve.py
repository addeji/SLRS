import matplotlib.pyplot as plt

def plot_learning_curves(history):
    """
    Plots the training and validation accuracy and loss over epochs.

    Args:
        history (tensorflow.keras.callbacks.History): The history object returned
                                                     by model.fit().
    """
    print("\n--- Plotting Learning Curves ---")
    plt.figure(figsize=(12, 5))

    # Plot training & validation accuracy values
    plt.subplot(1, 2, 1) # 1 row, 2 columns, 1st plot
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy over Epochs')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')
    plt.grid(True)

    # Plot training & validation loss values
    plt.subplot(1, 2, 2) # 1 row, 2 columns, 2nd plot
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss over Epochs')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    plt.grid(True)

    plt.tight_layout() # Adjusts plot parameters for a tight layout
    plt.show() # Display the plot
    print("Learning curves displayed.")

if __name__ == "__main__":
    # This block is for standalone testing if needed.
    print("This script is designed to be imported and called with a Keras training history object.")
    print("Example: from plot_learning_curves import plot_learning_curves")
    print("         plot_learning_curves(model_history)")
