# Data Science Portfolio

A collection of end-to-end data science projects covering **machine learning, statistical testing, mathematical optimization, and LLM applications**. Each project folder has its own README with the problem, approach, results, and how to run it.

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-LightGBM-2E7D32)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![PuLP](https://img.shields.io/badge/PuLP-Linear%20Programming-6A1B9A)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFD21E?logo=huggingface&logoColor=black)

---

## Featured projects

| Project | What it does | Key techniques | Highlights |
|---|---|---|---|
| [**Credit Risk Analysis**](Credit%20Risk%20Analysis/) | Predicts loan default probability and serves it through an interactive Streamlit app | Logistic Regression, XGBoost, LightGBM, SHAP | XGBoost test **AUC 0.95**; deployed web app with live scoring and explainability |
| [**RAG Data Chatbot**](AI%20Chatbot/) | Chat with any CSV file in plain English | Retrieval-Augmented Generation, ChromaDB, sentence-transformers, Qwen 2.5 LLMs | Streaming answers grounded in retrieved rows; upload-and-index from the UI |
| [**Marketing A/B Test**](AB_testing/) | Measures whether an ad campaign drove more conversions than a PSA control (588K users) | Welch's t-test, Chi-square, Bayesian Beta-Binomial, confidence intervals | Significant lift (p ≈ 2e-13); ~**4,300 conversions** attributed to ads |
| [**Supply Chain & Operations Optimization**](Optimization/) | Solves production, distribution, and scheduling decisions | Linear & mixed-integer programming (PuLP / CBC) | Multi-echelon network, production mix, and cinema showtime models |

## Additional projects

| Project | Summary |
|---|---|
| [**Titanic Survival (Kaggle)**](Kaggle%20Competition/Titanic_Disaster_Learning/) | Random Forest classifier with feature engineering and `GridSearchCV` hyperparameter tuning |
| [**Yu-Gi-Oh! Card Price Analysis**](Yugioh%20Forecast/) | Exploratory analysis of trading-card prices by rarity, plus a K-Fold cross-validated regression |
| [**Tree-Ensemble Feature Transformations**](Gradient%20Boost/) | Compares Random Forest / Gradient Boosting against leaf-embedding + Logistic Regression pipelines |
| [**Regression Model Examples**](Regression%20Model%20Examples/) | Reference notebooks for OLS, Ridge, Lasso, ElasticNet, Quantile, and Robust regression |
| [**Python & OOP Fundamentals**](Python/Object-Oriented-Functions/) | Progressive object-oriented programming exercises (bank system, state machines, games) |

---

## Skills demonstrated

- **Machine learning:** classification, regression, gradient boosting, ensembles, hyperparameter tuning, cross-validation
- **Model evaluation & explainability:** ROC/AUC, precision/recall, confusion matrices, feature importance, SHAP values
- **Statistics & experimentation:** hypothesis testing, frequentist vs. Bayesian A/B testing, confidence intervals
- **Optimization / Operations Research:** LP and MILP formulation, objective/constraint design, solver interpretation
- **Generative AI:** RAG pipelines, vector databases, text embeddings, LLM inference APIs, prompt design
- **Deployment:** interactive Streamlit apps, reproducible environments with Dev Containers / GitHub Codespaces

## Tech stack

`Python` · `pandas` · `NumPy` · `SciPy` · `scikit-learn` · `XGBoost` · `LightGBM` · `SHAP` · `PuLP` · `Matplotlib` · `Seaborn` · `Plotly` · `Streamlit` · `ChromaDB` · `sentence-transformers` · `Hugging Face Hub` · `Jupyter`

---

## Getting started

```bash
git clone https://github.com/mfearing1234/Data_science_model_builds.git
cd Data_science_model_builds
pip install -r requirements.txt
```

**Run the Credit Risk app**

```bash
streamlit run "Credit Risk Analysis/app.py"
```

**Run the RAG chatbot** (needs a free Hugging Face token — see the [project README](AI%20Chatbot/))

```bash
cd "AI Chatbot"
cp .env.example .env   # then add your HF_TOKEN
streamlit run app.py
```

**Open in GitHub Codespaces:** this repo includes a [Dev Container](.devcontainer/devcontainer.json) that installs all dependencies and launches the Credit Risk app automatically.

Datasets for most projects are pulled at runtime from Kaggle via [`kagglehub`](https://github.com/Kaggle/kagglehub); smaller datasets are included in their project folders.

## Repository structure

```
.
├── AI Chatbot/                  # RAG chatbot over CSV data (Streamlit + ChromaDB + HF)
├── AB_testing/                  # Marketing A/B test: frequentist & Bayesian
├── Credit Risk Analysis/        # Default prediction notebook + Streamlit app
├── Optimization/                # LP / MILP models with PuLP
├── Kaggle Competition/          # Titanic survival classifier
├── Yugioh Forecast/             # Card price EDA and regression
├── Gradient Boost/              # Tree-ensemble feature transformations
├── Regression Model Examples/   # Regression technique reference notebooks
├── Python/                      # OOP fundamentals
├── requirements.txt
└── .devcontainer/               # Codespaces / Dev Container config
```
