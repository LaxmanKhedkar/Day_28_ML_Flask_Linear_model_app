import pandas as pd
import numpy as np
import pickle
import config
import json

class MedicalInsurance():
    def __init__(self, age, sex, bmi, children, smoker, region):
        self.age = age
        self.sex = sex
        self.bmi = bmi
        self.children = children
        self.smoker = smoker
        self.region = "region_" + region

    def load_model(self):
        # Load trained model
        with open(config.MODEL_FILE_PATH, "rb") as file1:
            self.model = pickle.load(file1)

        # Load project data (columns, mappings)
        with open(config.PROJECT_DATA_PATH, "r") as file2:
            self.project_data = json.load(file2)

    def get_predicted_charges(self):
        self.load_model()

        # Create empty series with correct columns
        test_series = pd.Series(
            np.zeros(len(self.project_data["columns"])),
            index=self.project_data["columns"]
        )

        # Fill values
        test_series["age"] = self.age
        test_series["sex"] = self.project_data["sex"][self.sex]
        test_series["bmi"] = self.bmi
        test_series["children"] = self.children
        test_series["smoker"] = self.project_data["smoker"][self.smoker]

        if self.region in test_series.index:
            test_series[self.region] = 1  # One-hot encoding
        else:
            print(f"Warning: Region {self.region} not found in columns!")

        # Predict charges
        charges = self.model.predict([test_series.values])[0]
        return np.round(charges, 2)

if __name__ == "__main__":
    age = 19
    sex = "male"
    bmi = 28
    children = 0
    smoker = "yes"
    region = "southeast"

    med_insurance = MedicalInsurance(age, sex, bmi, children, smoker, region)
    charges = med_insurance.get_predicted_charges()

    print("********************************")
    print(f"The charges are: {charges}")
    print("********************************")
