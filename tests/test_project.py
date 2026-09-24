import json
import os

import joblib
import pandas as pd


def test_deployment_files_exist():
    assert os.path.exists("deployment/model.joblib")
    assert os.path.exists("deployment/model_metadata.json")
    assert os.path.exists("app.py")


def test_model_can_predict():
    model = joblib.load("deployment/model.joblib")
    with open("deployment/model_metadata.json", encoding="utf-8") as f:
        metadata = json.load(f)

    row = pd.DataFrame(
        [{
            "Age": 35,
            "TypeofContact": "Self Enquiry",
            "CityTier": 2,
            "DurationOfPitch": 15,
            "Occupation": "Salaried",
            "Gender": "Male",
            "NumberOfPersonVisiting": 2,
            "NumberOfFollowups": 3,
            "ProductPitched": "Basic",
            "PreferredPropertyStar": 4,
            "MaritalStatus": "Married",
            "NumberOfTrips": 3,
            "Passport": 1,
            "PitchSatisfactionScore": 3,
            "OwnCar": 1,
            "NumberOfChildrenVisiting": 0,
            "Designation": "Executive",
            "MonthlyIncome": 25000,
        }]
    )
    assert list(row.columns) == metadata["features"]
    probability = float(model.predict_proba(row)[:, 1][0])
    assert 0.0 <= probability <= 1.0
