import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

def train_and_save_model():
    # Load the data
    data = pd.read_csv('data/data.csv')

    # Preview the data
    print("📊 Data Overview:")
    print(data.head())

    # Feature columns and target column
    X = data.drop(columns='Recommended_Program')  # Features: All columns except the target
    y = data['Recommended_Program']  # Target: Recommended Program

    # Encode the target labels
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)  # Encoding the target labels (e.g., Technology -> 0, Arts -> 1)

    # Check the unique values in y_encoded
    print("Unique encoded labels:", np.unique(y_encoded))

    # Ensure the number of output classes matches the unique labels
    num_classes = len(np.unique(y_encoded))

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    # Build a simple Neural Network model using TensorFlow
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, input_dim=X_train.shape[1], activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')  # Adjust the number of output classes
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(X_train, y_train, epochs=100, batch_size=10, validation_data=(X_test, y_test))

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"\n✅ Model Training Complete! Accuracy: {accuracy:.2f}")

    # Predict with the model
    y_pred = np.argmax(model.predict(X_test), axis=1)  # Convert softmax output to class predictions

    # Ensure the number of unique labels matches the target names
    unique_labels = np.unique(y_encoded)
    target_names = encoder.inverse_transform(unique_labels)

    print("\n🔍 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names, zero_division=1))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

    # Save the model and encoder for later use
    model.save('model.h5')
    np.save('classes.npy', encoder.classes_)

def predict_program(scores):
    # Load the model and encoder
    model = tf.keras.models.load_model('model.h5')
    classes = np.load('classes.npy', allow_pickle=True)
    encoder = LabelEncoder()
    encoder.classes_ = classes

    # Make a prediction for the new student
    new_student_scores = np.array([scores])
    predicted_program = model.predict(new_student_scores)
    predicted_label = encoder.inverse_transform([np.argmax(predicted_program)])
    return predicted_label[0]

if __name__ == "__main__":
    train_and_save_model()