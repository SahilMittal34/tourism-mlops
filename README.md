# Visit with Us — Wellness Tourism MLOps

This repository predicts whether a customer will purchase the Wellness Tourism Package.

## Architecture

```text
Tourism CSV
    ↓
Data Preparation
    ↓
Train/Test Split
    ↓
Logistic Regression + Random Forest + Gradient Boosting
    ↓
Metric comparison + MLflow tracking
    ↓
Best model serialized with preprocessing
    ↓
GitHub Actions CI/CD
    ↓
Streamlit Community Cloud
```

The public frontend is hosted on **Streamlit Community Cloud**. Hugging Face is not required for deployment.

## Local run

```bash
pip install -r requirements.txt
python src/data_prep.py
python src/train.py
streamlit run app.py
```

## GitHub Actions

The workflow in `.github/workflows/pipeline.yml` runs on pushes to `main` and contains three stages:

1. `data-prep` — creates reproducible train/test artifacts.
2. `model-training` — trains Logistic Regression, Random Forest and Gradient Boosting, tracks experiments with MLflow, runs tests and produces the production model.
3. `deploy-streamlit` — validates the Streamlit application and commits the latest trained model artifact back to GitHub. Streamlit Community Cloud watches the GitHub repository and redeploys the application when the repository changes.

## Streamlit deployment

The entrypoint is:

```text
app.py
```

The model is loaded from:

```text
deployment/model.joblib
```

The application uses the same serialized scikit-learn preprocessing + model pipeline used during training, preventing a mismatch between training and inference transformations.

## Repository structure

```text
tourism-mlops/
├── .github/workflows/pipeline.yml
├── app.py
├── data/tourism.csv
├── src/
│   ├── data_prep.py
│   └── train.py
├── model_building/
│   ├── best_model.joblib
│   └── experiment_results.csv
├── deployment/
│   ├── model.joblib
│   └── model_metadata.json
├── artifacts/
│   ├── train.csv
│   └── test.csv
├── tests/test_project.py
├── requirements.txt
└── requirements-actions.txt
```

## Final project links

- GitHub repository: `PASTE_YOUR_GITHUB_REPOSITORY_URL_HERE`
- Streamlit application: `PASTE_YOUR_STREAMLIT_APP_URL_HERE`
