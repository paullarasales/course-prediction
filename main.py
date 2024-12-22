from predict import predict_course
import warnings

warnings.filterwarnings("ignore")

def main():
    print("Welcome to the Course Recommendation System!")

    # Ask the user for their scores
    technology_score = int(input("Enter your Technology score (1-10): "))
    science_score = int(input("Enter your Science score (1-10): "))
    arts_score = int(input("Enter your Arts score (1-10): "))
    business_score = int(input("Enter your Business score (1-10): "))

    # Prepare the user input as a list
    user_scores = [technology_score, science_score, arts_score, business_score]

    # Get the recommended course
    recommended_course = predict_course(user_scores)
    
    # Show the recommendation to the user
    print(f"\nRecommended Program: {recommended_course}")

if __name__ == "__main__":
    main()
