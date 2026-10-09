# Titanic Survival Prediction (Kaggle)

Entry for Kaggle's classic [Titanic — Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic) competition: predict which passengers survived the sinking based on ticket class, age, fare, family size, and other attributes.

## Data

| File | Description |
|---|---|
| `train.csv` | 891 passengers with the `Survived` label |
| `test.csv` | 418 passengers to predict (Kaggle submission set) |

## Approach

1. **Feature engineering** — normalized passenger names and split ticket strings into a ticket prefix and ticket number; experimented with Keras `Tokenizer` encodings of text fields.
2. **Baseline model** — `RandomForestClassifier` (200 trees, max depth 5) evaluated with the **out-of-bag (OOB) score**: **0.73**.
3. **Hyperparameter tuning** — `GridSearchCV` across 72 combinations of `max_depth`, `min_samples_leaf`, and `n_estimators`, optimized for accuracy.
4. **Interpretation** — visualized an individual decision tree and ranked feature importances. **Age** and **Fare** were the strongest predictors.
5. **Prediction** — generated survival predictions for the test set.

## How to run

```bash
pip install -r ../../requirements.txt tensorflow
jupyter notebook Titanic_Disaster_RF_Model.ipynb
```

(`tensorflow` is only needed for the Keras tokenizer experiment.)

## Lessons learned / next steps

This was one of my earliest projects. Looking back, the biggest improvements would be:

- **Impute instead of dropping rows** — `dropna()` removes most passengers because `Cabin` is usually missing; imputing `Age` and dropping only `Cabin` would keep the full dataset.
- **Encode `Sex` and `Pclass` as features** — passenger sex is historically the single strongest predictor of survival.
- **Exclude `PassengerId`** — it is an identifier, not a signal.
