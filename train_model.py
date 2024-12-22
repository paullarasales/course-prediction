import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard
import os
import joblib

def train_and_save_model():
    # Load the dataset
    data = pd.read_csv("data/large_course_recommendation_dataset.csv")

    # Feature columns
    X = data[['Technology_Score', 'Science_Score', 'Arts_Score', 'Business_Score']]
    y = data['Recommended_Program']

    # Encode the labels since TensorFlow models need numeric labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    # Build the neural network model
    model = Sequential()
    model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))  # Input layer and hidden layer
    model.add(Dense(32, activation='relu'))  # hidden layer
    model.add(Dense(len(label_encoder.classes_), activation='softmax'))  # Output layer (softmax for multi-class classification)
    
    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    log_dir = os.path.join("logs", "fit")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    # Train the model
    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test), callbacks=[tensorboard_callback])

    # Evaluate the model
    _, accuracy = model.evaluate(X_test, y_test)
    print(f"Model accuracy: {accuracy * 100:.2f}%")

    # Save the trained model and label encoder
    model.save('course_recommendation_model.h5')
    joblib.dump(label_encoder, 'label_encoder.pkl')
    print("Model trained and saved as 'course_recommendation_model.h5'.")
    print("Label encoder saved as 'label_encoder.pkl'.")

# Run the training function
if __name__ == "__main__":
    train_and_save_model()
