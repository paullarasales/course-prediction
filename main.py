import csv
import os
from train_model import predict_program

def get_valid_score(question):
    """Ask a question and validate that the input is a number between 1 and 10."""
    while True:
        try:
            score = int(input(question))
            if 1 <= score <= 10:
                return score
            else:
                print("Please enter a number between 1 and 10.")
        except ValueError:
            print("Invalid input. Please enter a valid number between 1 and 10.")

def save_to_csv(data, filename="data.csv"):
    """Save data (dictionary) to a CSV file in the data folder."""
    data_folder = "data"
    os.makedirs(data_folder, exist_ok=True)

    filepath = os.path.join(data_folder, filename)
   
    with open(filepath, mode="a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=data.keys())
        if file.tell() == 0: 
            writer.writeheader()
        writer.writerow(data)

def entrance_exam():
    print("=" * 50)
    print("🎓 Welcome to the Student Program Recommendation System 🎓")
    print("Please answer the following questions on a scale of 1 to 10.\n")
    
    tech_score = get_valid_score("How much do you like technology? ")
    science_score = get_valid_score("How much do you like science? ")
    arts_score = get_valid_score("How much do you like arts? ")
    business_score = get_valid_score("How much do you like business? ")

    scores = {
        "Technology_Score": tech_score,
        "Science_Score": science_score,
        "Arts_Score": arts_score,
        "Business_Score": business_score
    }
    recommended_program = predict_program([tech_score, science_score, arts_score, business_score])

    student_data = {
        "Technology_Score": tech_score,
        "Science_Score": science_score,
        "Arts_Score": arts_score,
        "Business_Score": business_score,
        "Recommended_Program": recommended_program
    }
    save_to_csv(student_data)
    print("\n" + "=" * 50)
    print("🎉 Based on your answers 🎉")
    print(f"👉 Recommended Program: {recommended_program}")
    print("Your answers have been saved. Thank you!")
    print("=" * 50)

if __name__ == "__main__":
    entrance_exam()