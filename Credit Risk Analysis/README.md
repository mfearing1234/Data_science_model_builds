# Credit Risk Analysis — Loan Default Prediction

Predicts the probability that a borrower will default on a loan, compares three classification models, and serves the results in an interactive **Streamlit** web app with live scoring and model explainability.

## Business context

Lenders estimate **Expected Credit Loss** as:

> **ECL = PD × LGD × EAD**
> (Probability of Default × Loss Given Default × Exposure at Default)

This project focuses on the **PD** component — a binary classifier that scores each applicant's likelihood of default.

## Data

[Credit Risk Dataset (Kaggle)](https://www.kaggle.com/datasets/laotse/credit-risk-dataset) — ~32,500 loans with applicant demographics, income, employment length, loan amount/grade/intent, interest rate, and credit-bureau history. Target: `loan_status` (1 = default, ~22% of loans).

Downloaded automatically at runtime via `kagglehub`.

## Approach

1. **Data cleaning** — profiled nulls (`person_emp_length`: 895, `loan_int_rate`: 3,116) and imputed missing values.
2. **Feature engineering** — one-hot encoded loan grade, home ownership, loan intent, and prior-default flag.
3. **Modeling** — stratified 80/20 train/test split, then:
   - **Logistic Regression** baseline, validated with 5-fold cross-validation
   - **XGBoost** with a grid search over learning rate and number of trees
   - **LightGBM** for comparison
4. **Explainability** — XGBoost feature importance and **SHAP** summary plots to show which factors drive default risk.

## Results

| Model | Test AUC-ROC |
|---|---|
| Logistic Regression | 0.847 (5-fold CV mean: 0.849) |
| **XGBoost** (lr = 0.1, 300 trees) | **0.952** |
| LightGBM | ROC curve in notebook |

Tree-based models substantially outperform the linear baseline, capturing non-linear interactions between income, loan-to-income ratio, interest rate, and loan grade.

## Streamlit app (`app.py`)

| Page | What you see |
|---|---|
| **Data Overview** | Record count, default rate, raw data sample, null profile, class balance |
| **Model Performance** | Side-by-side AUC metrics, overlaid ROC curves, classification reports, confusion matrices |
| **Predict Default Risk** | Enter an applicant's details and get default probabilities from all three models plus an ensemble average |
| **Feature Importance** | Interactive top-N XGBoost feature importance chart |
| **SHAP Values** | SHAP bar and beeswarm plots on an adjustable sample size |

## How to run

```bash
pip install -r ../requirements.txt

# Interactive app
streamlit run app.py

# Or explore the analysis
jupyter notebook credit_risk_analysis.ipynb
```

## Files

| File | Description |
|---|---|
| `credit_risk_analysis.ipynb` | Full analysis: cleaning, modeling, evaluation, SHAP |
| `app.py` | Streamlit application |

## Possible next steps

- Tune hyperparameters with cross-validation on the training set only, keeping the test set fully held out
- Address the class imbalance (class weights or threshold tuning) to improve default recall
- Cap outliers such as implausible employment lengths and ages
- Extend from PD to a full ECL estimate by modeling LGD and EAD
