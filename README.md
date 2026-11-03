# Tumor Classification — End-to-End Machine Learning Project

A beginner-friendly but portfolio-ready **binary classification** project using the Wisconsin Diagnostic Breast Cancer dataset built into scikit-learn.

The project trains a Logistic Regression model to classify a tumor measurement profile as **benign** or **malignant**. It includes a reproducible training pipeline, evaluation metrics, saved model, and an interactive Streamlit app.

> **Important:** This is an educational machine-learning demonstration, not a medical diagnostic tool. Never use its predictions for healthcare decisions.

## Features

- Clean train/test split with stratification
- StandardScaler + Logistic Regression in a single scikit-learn Pipeline
- Accuracy, precision, recall, F1-score, ROC-AUC
- Confusion matrix and ROC curve
- Saved model and evaluation report
- Streamlit interface for interactive predictions
- Reproducible random seed

## Project structure

```text
ML_Project_Customer_Tumor_Classification/
├── app.py
├── train.py
├── requirements.txt
├── README.md
├── .gitignore
├── src/
│   ├── __init__.py
│   └── ml_pipeline.py
├── models/                 # generated after training
└── reports/                # generated after training
```

## 1. Set up the environment

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

## 2. Train the model

```bash
python train.py
```

This creates:
- `models/tumor_classifier.joblib`
- `reports/metrics.json`
- `reports/classification_report.txt`

## 3. Run the interactive app

```bash
streamlit run app.py
```

Open the local URL printed in your terminal (usually http://localhost:8501).

## Dataset

The project uses `sklearn.datasets.load_breast_cancer`, a built-in educational dataset containing 569 samples and 30 numeric features. The target is binary:
- `0` — malignant
- `1` — benign

The app asks for the 30 measurements. To keep the interface manageable, it provides a sample from the dataset as a starting point and allows individual values to be edited.

## Model choice

Logistic Regression is a useful baseline for binary classification. Features are standardized before training because they have different numeric scales. The pipeline keeps preprocessing and prediction together, reducing the risk of inconsistent transformations.

## Evaluation

The test set is held out from training. The script reports:
- Accuracy: overall fraction of correct predictions
- Precision: fraction of predicted positive cases that are correct
- Recall: fraction of actual positive cases detected
- F1-score: harmonic mean of precision and recall
- ROC-AUC: ranking quality across classification thresholds

For a medical dataset, recall and false negatives deserve particular attention. Metrics from this educational dataset do not establish real-world clinical performance.

## GitHub upload

```bash
git init
git add .
git commit -m "Add end-to-end tumor classification ML project"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
git push -u origin main
```

Replace the remote URL with your own GitHub repository URL.

## Possible improvements

1. Compare Logistic Regression, Random Forest, and Support Vector Machine.
2. Add cross-validation and hyperparameter tuning.
3. Add feature importance or permutation importance.
4. Add unit tests and a GitHub Actions workflow.
5. Experiment with a decision threshold and explain precision/recall trade-offs.
