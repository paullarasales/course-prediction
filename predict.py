import joblib

def predict_course(user_scores):
    # Load the saved model
    model = joblib.load('course_recommendation_model.pkl')

    # Predict the course based on the user scores
    predicted_course = model.predict([user_scores])
    
    return predicted_course[0]

