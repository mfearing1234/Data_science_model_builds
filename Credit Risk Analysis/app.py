import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import os

st.set_page_config(page_title="Credit Risk Analysis", layout="wide")
st.title("Credit Risk Analysis Model Exploration")
st.write("This app allows you to explore the performance of different models for credit risk analysis.")


@st.cache_data
def load_data():
    import kagglehub
    path = kagglehub.dataset_download("laotse/credit-risk-dataset")
    files = os.listdir(path)
    df = pd.read_csv(os.path.join(path, files[0]))
    return df


@st.cache_data
def preprocess(df):
    df = df.copy()
    df.fillna(0, inplace=True)
    encoded = pd.get_dummies(
        df,
        columns=["loan_grade", "person_home_ownership", "loan_intent", "cb_person_default_on_file"],
        drop_first=True,
    )
    return encoded


@st.cache_data
def train_models(df_encoded):
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import roc_auc_score, roc_curve, classification_report, confusion_matrix
    from xgboost import XGBClassifier
    import lightgbm as lgb

    X = df_encoded.drop("loan_status", axis=1)
    y = df_encoded["loan_status"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Logistic Regression (scaled)
    lr_model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000)),
    ])
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)
    y_prob_lr = lr_model.predict_proba(X_test)[:, 1]
    auc_lr = roc_auc_score(y_test, y_prob_lr)
    fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)

    # XGBoost
    xgb_model = XGBClassifier(eval_metric="logloss", learning_rate=0.1, n_estimators=200)
    xgb_model.fit(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_test)
    y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]
    auc_xgb = roc_auc_score(y_test, y_prob_xgb)
    fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_prob_xgb)

    # LightGBM
    lgbm_model = lgb.LGBMClassifier(learning_rate=0.1, n_estimators=200, verbosity=-1)
    lgbm_model.fit(X_train, y_train)
    y_pred_lgbm = lgbm_model.predict(X_test)
    y_prob_lgbm = lgbm_model.predict_proba(X_test)[:, 1]
    auc_lgbm = roc_auc_score(y_test, y_prob_lgbm)
    fpr_lgbm, tpr_lgbm, _ = roc_curve(y_test, y_prob_lgbm)

    return {
        "X": X, "y": y,
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "lr_model": lr_model, "y_pred_lr": y_pred_lr,
        "auc_lr": auc_lr, "fpr_lr": fpr_lr, "tpr_lr": tpr_lr,
        "xgb_model": xgb_model, "y_pred_xgb": y_pred_xgb,
        "auc_xgb": auc_xgb, "fpr_xgb": fpr_xgb, "tpr_xgb": tpr_xgb,
        "lgbm_model": lgbm_model, "y_pred_lgbm": y_pred_lgbm,
        "auc_lgbm": auc_lgbm, "fpr_lgbm": fpr_lgbm, "tpr_lgbm": tpr_lgbm,
        "report_lr": classification_report(y_test, y_pred_lr, output_dict=True),
        "report_xgb": classification_report(y_test, y_pred_xgb, output_dict=True),
        "report_lgbm": classification_report(y_test, y_pred_lgbm, output_dict=True),
        "cm_lr": confusion_matrix(y_test, y_pred_lr),
        "cm_xgb": confusion_matrix(y_test, y_pred_xgb),
        "cm_lgbm": confusion_matrix(y_test, y_pred_lgbm),
        "report_lr_text": classification_report(y_test, y_pred_lr),
        "report_xgb_text": classification_report(y_test, y_pred_xgb),
        "report_lgbm_text": classification_report(y_test, y_pred_lgbm),
    }


# --- Load & preprocess ---
with st.spinner("Loading dataset..."):
    df_raw = load_data()
    df_encoded = preprocess(df_raw)

# --- Sidebar ---
st.sidebar.header("Navigation")
section = st.sidebar.radio(
    "Section",
    ["Data Overview", "Model Performance", "Predict Default Risk", "Feature Importance", "SHAP Values"],
)

# --- Train models (cached) ---
with st.spinner("Training models (first run only)..."):
    results = train_models(df_encoded)

# ── Data Overview ──────────────────────────────────────────────────────────────
if section == "Data Overview":
    st.header("Data Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", f"{len(df_raw):,}")
    col2.metric("Default Rate", f"{df_raw['loan_status'].mean():.1%}")
    col3.metric("Features", len(df_raw.columns) - 1)

    st.subheader("Raw Data Sample")
    st.dataframe(df_raw.head(50), use_container_width=True)

    st.subheader("Null Values (before cleaning)")
    null_counts = df_raw.isnull().sum()
    null_df = null_counts[null_counts > 0].reset_index()
    null_df.columns = ["Column", "Null Count"]
    st.dataframe(null_df, use_container_width=True)

    st.subheader("Loan Status Distribution")
    counts = df_raw["loan_status"].value_counts().rename({0: "No Default", 1: "Default"})
    fig = go.Figure(go.Bar(x=counts.index, y=counts.values, marker_color=["steelblue", "tomato"]))
    fig.update_layout(xaxis_title="Loan Status", yaxis_title="Count")
    st.plotly_chart(fig, use_container_width=True)

# ── Model Performance ──────────────────────────────────────────────────────────
elif section == "Model Performance":
    st.header("Model Performance Metrics")

    col1, col2, col3 = st.columns(3)
    col1.metric("Logistic Regression AUC", f"{results['auc_lr']:.4f}")
    col2.metric("XGBoost AUC", f"{results['auc_xgb']:.4f}")
    col3.metric("LightGBM AUC", f"{results['auc_lgbm']:.4f}")

    st.subheader("ROC Curve Comparison")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=results["fpr_lr"], y=results["tpr_lr"],
        name=f"Logistic Regression (AUC={results['auc_lr']:.2f})", mode="lines",
    ))
    fig.add_trace(go.Scatter(
        x=results["fpr_xgb"], y=results["tpr_xgb"],
        name=f"XGBoost (AUC={results['auc_xgb']:.2f})", mode="lines",
    ))
    fig.add_trace(go.Scatter(
        x=results["fpr_lgbm"], y=results["tpr_lgbm"],
        name=f"LightGBM (AUC={results['auc_lgbm']:.2f})", mode="lines",
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        line=dict(dash="dash", color="gray"), name="Random",
    ))
    fig.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Classification Reports")
    tab1, tab2, tab3 = st.tabs(["Logistic Regression", "XGBoost", "LightGBM"])
    for tab, key in [(tab1, "lr"), (tab2, "xgb"), (tab3, "lgbm")]:
        with tab:
            st.text(results[f"report_{key}_text"])
            cm = results[f"cm_{key}"]
            st.dataframe(
                pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"]),
                use_container_width=True,
            )

# ── Predict Default Risk ───────────────────────────────────────────────────────
elif section == "Predict Default Risk":
    st.header("Predict Default Risk")

    col1, col2 = st.columns(2)
    with col1:
        person_age = st.number_input("Person Age", min_value=18, max_value=100, value=30)
        person_income = st.number_input("Person Income", min_value=0, value=50000)
        person_emp_length = st.number_input("Employment Length (years)", min_value=0, max_value=50, value=5)
        cb_person_cred_hist_length = st.number_input("Credit History Length (years)", min_value=0, max_value=50, value=5)
        cb_person_default_on_file = st.selectbox("Prior Default on File", options=["N", "Y"])
    with col2:
        loan_amnt = st.number_input("Loan Amount", min_value=0, value=10000)
        loan_int_rate = st.number_input("Loan Interest Rate (%)", min_value=0.0, max_value=100.0, value=10.0)
        loan_grade = st.selectbox("Loan Grade", options=["A", "B", "C", "D", "E", "F", "G"])
        person_home_ownership = st.selectbox("Home Ownership", options=["MORTGAGE", "OTHER", "OWN", "RENT"])
        loan_intent = st.selectbox("Loan Intent", options=["DEBTCONSOLIDATION", "EDUCATION", "HOMEIMPROVEMENT", "MEDICAL", "PERSONAL", "VENTURE"])

    # loan_percent_income is a derived feature the model was trained on
    loan_percent_income = loan_amnt / person_income if person_income > 0 else 0.0

    # Columns must match exactly what drop_first=True produces from training data:
    # base categories (dropped): loan_grade=A, home_ownership=MORTGAGE,
    #                             loan_intent=DEBTCONSOLIDATION, cb_default=N
    input_data = pd.DataFrame({
        "person_age": [person_age],
        "person_income": [person_income],
        "person_emp_length": [person_emp_length],
        "loan_amnt": [loan_amnt],
        "loan_int_rate": [loan_int_rate],
        "loan_percent_income": [loan_percent_income],
        "cb_person_cred_hist_length": [cb_person_cred_hist_length],
        "loan_grade_B": [int(loan_grade == "B")],
        "loan_grade_C": [int(loan_grade == "C")],
        "loan_grade_D": [int(loan_grade == "D")],
        "loan_grade_E": [int(loan_grade == "E")],
        "loan_grade_F": [int(loan_grade == "F")],
        "loan_grade_G": [int(loan_grade == "G")],
        "person_home_ownership_OTHER": [int(person_home_ownership == "OTHER")],
        "person_home_ownership_OWN": [int(person_home_ownership == "OWN")],
        "person_home_ownership_RENT": [int(person_home_ownership == "RENT")],
        "loan_intent_EDUCATION": [int(loan_intent == "EDUCATION")],
        "loan_intent_HOMEIMPROVEMENT": [int(loan_intent == "HOMEIMPROVEMENT")],
        "loan_intent_MEDICAL": [int(loan_intent == "MEDICAL")],
        "loan_intent_PERSONAL": [int(loan_intent == "PERSONAL")],
        "loan_intent_VENTURE": [int(loan_intent == "VENTURE")],
        "cb_person_default_on_file_Y": [int(cb_person_default_on_file == "Y")],
    })

    st.divider()
    xgb_prob = results["xgb_model"].predict_proba(input_data)[:, 1][0]
    lgbm_prob = results["lgbm_model"].predict_proba(input_data)[:, 1][0]
    lr_prob = results["lr_model"].predict_proba(input_data)[:, 1][0]

    c1, c2, c3 = st.columns(3)
    c1.metric("XGBoost Default Probability", f"{xgb_prob:.1%}")
    c2.metric("LightGBM Default Probability", f"{lgbm_prob:.1%}")
    c3.metric("Logistic Regression Default Probability", f"{lr_prob:.1%}")

    avg_prob = (xgb_prob + lgbm_prob + lr_prob) / 3
    risk_label = "High Risk" if avg_prob >= 0.5 else "Low Risk"
    color = "🔴" if avg_prob >= 0.5 else "🟢"
    st.markdown(f"### {color} Ensemble Average: **{avg_prob:.1%}** — {risk_label}")

# ── Feature Importance ─────────────────────────────────────────────────────────
elif section == "Feature Importance":
    st.header("XGBoost Feature Importance")

    xgb_model = results["xgb_model"]
    X = results["X"]
    importance = pd.Series(xgb_model.feature_importances_, index=X.columns)
    top_n = st.slider("Top N features", min_value=5, max_value=len(importance), value=10)
    top = importance.nlargest(top_n).sort_values()

    fig = go.Figure(go.Bar(
        x=top.values, y=top.index, orientation="h", marker_color="steelblue"
    ))
    fig.update_layout(
        xaxis_title="Importance Score",
        yaxis_title="Feature",
        height=max(300, top_n * 30),
    )
    st.plotly_chart(fig, use_container_width=True)

# ── SHAP Values ────────────────────────────────────────────────────────────────
elif section == "SHAP Values":
    st.header("SHAP Values (XGBoost)")

    sample_size = st.slider(
        "Sample size for SHAP (larger = slower)", min_value=100, max_value=2000, value=500, step=100
    )

    with st.spinner("Computing SHAP values..."):
        import shap
        X_test = results["X_test"]
        xgb_model = results["xgb_model"]
        X_sample = X_test.sample(min(sample_size, len(X_test)), random_state=42)
        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(X_sample)

    st.subheader("SHAP Summary Plot (Bar)")
    fig, _ = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
    st.pyplot(fig)
    plt.close()

    st.subheader("SHAP Beeswarm Plot")
    fig2, _ = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values, X_sample, show=False)
    st.pyplot(fig2)
    plt.close()

