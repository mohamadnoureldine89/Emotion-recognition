# performance_evaluation.py

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def plot_accuracy_loss(history):
    """
    Plot training and validation accuracy and loss over epochs.
    Args:
        history (tf.keras.callbacks.History): History object returned from model training.
    """
    plt.figure(figsize=(12, 4))

    # Plot training and validation accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training Accuracy vs Validation Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')

    # Plot training and validation loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training Loss vs Validation Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')

    plt.show()

def evaluate_model(model, data):
    """
    Evaluate the performance of the trained model on the validation dataset.
    Args:
        model (tf.keras.models.Model): Trained machine learning model.
        data: Validation data (features and labels).
    """
    X_val, y_val = data  # Split data into features and labels

    # Predict labels on the validation data
    y_pred = model.predict(X_val)

    # Convert predicted probabilities to class labels
    y_pred_labels = np.argmax(y_pred, axis=1)

    # Calculate accuracy
    accuracy = accuracy_score(np.argmax(y_val, axis=1), y_pred_labels)
    print(f"Validation Accuracy: {accuracy:.2f}")

    # Generate a confusion matrix
    conf_matrix = confusion_matrix(np.argmax(y_val, axis=1), y_pred_labels)
    print("Confusion Matrix:")
    print(conf_matrix)

    # Generate a classification report
    class_report = classification_report(np.argmax(y_val, axis=1), y_pred_labels)
    print("Classification Report:")
    print(class_report)

if __name__ == "__main__":
    # You can include test code here to run the module independently if needed.
    pass
