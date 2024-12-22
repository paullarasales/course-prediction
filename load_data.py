import pandas as pd
import os

def load_data():
    """Load data from the CSV file and display it."""
    data_folder = "data"
    filepath = os.path.join(data_folder, "data.csv")
    print("Absolute path of the file:", os.path.abspath(filepath)) 

    try:
        df = pd.read_csv(filepath)
        print("📊 Data Loaded Successfully!\n")
        print(df)
    except FileNotFoundError:
        print("❌ Error: data.csv file not found. Please ensure the file is in the 'data' folder.")

if __name__ == "__main__":
    load_data()
