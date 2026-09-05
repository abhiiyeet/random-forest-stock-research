"""Minimal Streamlit interface for the Random Forest research workflow."""

from __future__ import annotations

import hashlib

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from export_utils import (
    create_excel_report,
    create_research_package,
    dataframe_csv,
    metrics_frame,
    model_bytes,
)
from ml_pipeline import (
    DEFAULT_PARAMS,
    FEATURE_LABELS,
    DataValidationError,
    clean_stock_data,
    evaluate_model,
    prepare_data,
    train_model,
)


st.set_page_config(page_title="Random Forest Stock Prediction", page_icon="🌲", layout="centered")
st.markdown("""
<style>
    .block-container {max-width: 980px; padding-top: 2.7rem; padding-bottom: 4rem;}
    h1 {color:#203746; letter-spacing:-0.025em;}
    h2, h3 {color:#29495e;}
    .stButton button[kind="primary"] {background:#3478a8; border-color:#3478a8;}
    div[data-testid="stMetric"] {background:#f7f9fa; border:1px solid #e5eaed; padding:0.8rem; border-radius:0.35rem;}
    .muted {color:#657780; margin-top:-0.65rem; margin-bottom:2rem;}
</style>
""", unsafe_allow_html=True)

st.title("Random Forest Stock Prediction")
st.markdown('<div class="muted">Upload historical daily stock data, train the model, and evaluate next-day UP/DOWN predictions.</div>', unsafe_allow_html=True)

st.header("1. Upload Data")
training_file = st.file_uploader(
    "Upload Stock Data (CSV or Excel workbook)",
    type=["csv", "xlsx", "xlsm", "xls"],
    key="training_file",
    help="For Excel workbooks, every non-empty worksheet is imported and combined chronologically.",
)
test_file = st.file_uploader(
    "Optional: Upload Separate Test-Year File",
    type=["csv", "xlsx", "xlsm", "xls"],
    key="test_file",
)

if training_file:
    try:
        clean_preview = clean_stock_data(training_file)
        training_file.seek(0)
        st.success("Data Loaded Successfully")
        st.caption(
            f"Period: {clean_preview.Date.min():%Y-%m-%d} → {clean_preview.Date.max():%Y-%m-%d}  ·  "
            f"Rows: {len(clean_preview):,}  ·  Latest Year: {clean_preview.Date.dt.year.max()}"
        )
        if clean_preview.Date.dt.year.nunique() < 4 and test_file is None:
            st.warning("This dataset contains a relatively short historical period. Results may be less stable.")
        last_date = clean_preview.Date.max()
        if test_file is None and (last_date.month < 11 or len(clean_preview[clean_preview.Date.dt.year == last_date.year]) < 180):
            st.info("The latest calendar year appears to contain only part of the year. Evaluation will use the available observations.")
    except DataValidationError as exc:
        training_file.seek(0)
        st.error(str(exc))

st.header("2. Select Features")
selected_features = []
columns = st.columns(2)
for index, (feature, label) in enumerate(FEATURE_LABELS.items()):
    if columns[index % 2].checkbox(label, value=True, key=f"feature_{feature}"):
        selected_features.append(feature)

with st.expander("Advanced Settings"):
    left, right = st.columns(2)
    n_estimators = left.number_input("Number of trees", 50, 2000, 300, 50)
    max_depth = right.number_input("Maximum depth", 1, 50, 6)
    min_samples_split = left.number_input("Minimum samples split", 2, 200, 20)
    min_samples_leaf = right.number_input("Minimum samples leaf", 1, 100, 10)
    max_features = left.selectbox("Maximum features", ["sqrt", "log2", None], index=0)
    random_state = right.number_input("Random state", 0, 1_000_000, 42)
    transaction_cost = st.number_input("Transaction Cost Per Position Change (%)", 0.0, 10.0, 0.0, 0.01, format="%.2f")

parameters = {
    **DEFAULT_PARAMS,
    "n_estimators": int(n_estimators), "max_depth": int(max_depth),
    "min_samples_split": int(min_samples_split), "min_samples_leaf": int(min_samples_leaf),
    "max_features": max_features, "random_state": int(random_state),
}

signature_parts = [training_file.getvalue() if training_file else b"", test_file.getvalue() if test_file else b"", repr(selected_features).encode(), repr(parameters).encode(), repr(transaction_cost).encode()]
current_signature = hashlib.sha256(b"".join(signature_parts)).hexdigest()
if st.session_state.get("signature") not in (None, current_signature):
    for key in ["model", "prepared", "results", "parameters"]:
        st.session_state.pop(key, None)

if st.button("Train Model", type="primary", use_container_width=True):
    try:
        if training_file is None:
            raise DataValidationError("Please upload a historical stock CSV or Excel workbook before training.")
        if not selected_features:
            raise DataValidationError("At least one feature must be selected.")
        training_file.seek(0)
        if test_file: test_file.seek(0)
        prepared = prepare_data(training_file, selected_features, test_file)
        model, fitted_parameters = train_model(prepared, parameters)
        st.session_state.update(model=model, prepared=prepared, parameters=fitted_parameters, signature=current_signature)
        st.session_state.pop("results", None)
    except DataValidationError as exc:
        st.error(str(exc))
    except Exception:
        st.error("The model could not be trained. Please check the uploaded data and settings.")

if "model" in st.session_state:
    prepared = st.session_state.prepared
    st.success("Model Trained Successfully")
    a, b, c = st.columns(3)
    a.metric("Training Period", prepared.training_period)
    b.metric("Training Rows", f"{len(prepared.train_data):,}")
    c.metric("Features Used", len(prepared.selected_features))
    st.caption(f"Prediction/Test Period: {prepared.test_period} · Raw Rows: {prepared.raw_rows:,} · Usable Rows: {prepared.usable_rows:,}")

predict_disabled = "model" not in st.session_state
if st.button("Predict & Compare", type="primary", use_container_width=True, disabled=predict_disabled):
    try:
        st.session_state.results = evaluate_model(st.session_state.model, st.session_state.prepared, transaction_cost)
    except Exception:
        st.error("Predictions could not be generated from the held-out test data.")

if "results" in st.session_state:
    results = st.session_state.results
    prepared = st.session_state.prepared
    st.divider()
    st.header("Model Performance")
    metric_cols = st.columns(5)
    for col, name in zip(metric_cols, ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]):
        value = results["metrics"][name]
        display = "N/A" if value is None else (f"{value:.1%}" if name != "ROC-AUC" else f"{value:.3f}")
        col.metric("F1 Score" if name == "F1" else name, display)

    latest = results["latest_prediction"]
    if latest:
        st.subheader("Latest Next-Day Prediction")
        a, b, c = st.columns(3)
        a.metric("Date", latest["Date"].strftime("%d-%m-%Y")); b.metric("Prediction", "UP" if latest["Prediction"] else "DOWN"); c.metric("Probability of UP", f"{latest['Probability_UP']:.1%}")
        st.caption("Unverified prediction — actual next-day movement is not available in the uploaded data.")

    st.subheader("Predicted vs Actual")
    predictions = results["predictions"]
    correct = int(predictions["Correct"].sum())
    a, b, c = st.columns(3); a.metric("Correct Predictions", correct); b.metric("Incorrect Predictions", len(predictions)-correct); c.metric("Accuracy", f"{correct/len(predictions):.2%}")
    display = predictions[["Date", "Close", "Actual_Direction", "Predicted_Direction", "Probability_UP", "Correct"]].rename(columns={"Actual_Direction":"Actual", "Predicted_Direction":"Predicted", "Probability_UP":"UP Probability"})
    st.dataframe(display.style.format({"Date": lambda x: x.strftime("%d-%m-%Y"), "Close":"{:,.2f}", "UP Probability":"{:.1%}"}), use_container_width=True, hide_index=True, height=360)

    st.subheader("Confusion Matrix")
    cm = results["confusion_matrix"]
    st.dataframe(pd.DataFrame(cm, index=["Actual DOWN", "Actual UP"], columns=["Predicted DOWN", "Predicted UP"]), use_container_width=True)

    st.subheader("Feature Importance")
    importance = results["feature_importance"].sort_values("Importance")
    fig, ax = plt.subplots(figsize=(7, 3.2)); ax.barh(importance["Display Feature"], importance["Importance"], color="#3478a8"); ax.set_xlabel("Relative importance"); ax.spines[["top", "right"]].set_visible(False); fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    st.caption("Random Forest feature importance indicates the model's relative use of each feature and should not be interpreted as proof of causation.")

    st.subheader("Random Forest vs Baseline")
    benchmark = results["benchmark_comparison"].copy()
    st.dataframe(benchmark.style.format({"Random Forest":"{:.1%}", "Naive Baseline":"{:.1%}", "Difference":"{:+.1%}"}), use_container_width=True, hide_index=True)

    st.subheader("Trading Strategy vs Buy-and-Hold")
    trading_table = pd.DataFrame(results["trading_metrics"]).reset_index().rename(columns={"index":"Metric"})
    def trade_format(row):
        return "{:.3f}" if row == "Sharpe Ratio" else "{:.2%}"
    formatted = trading_table.copy()
    for i, row in formatted.iterrows():
        for col in ["RF Strategy", "Buy & Hold"]: formatted.loc[i, col] = trade_format(row["Metric"]).format(row[col])
    st.dataframe(formatted, use_container_width=True, hide_index=True)
    trading = results["trading_performance"].set_index("Date")[["RF Cumulative Value", "Buy-and-Hold Cumulative Value"]]
    st.line_chart(trading, color=["#3478a8", "#8b969d"])
    st.caption(f"Close-to-close framework · Transaction cost per position change: {transaction_cost:.2f}% · Risk-free rate assumed: 0% · 252 trading days per year.")

    st.divider()
    st.header("Export Results")
    year = prepared.test_data.Date.max().year
    excel = create_excel_report(prepared, results, st.session_state.parameters, transaction_cost)
    package = create_research_package(prepared, results, st.session_state.model, st.session_state.parameters, transaction_cost)
    col1, col2 = st.columns(2)
    col1.download_button("Download Excel Report", excel, f"random_forest_stock_research_{year}.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
    col2.download_button("Download Predictions CSV", dataframe_csv(predictions), "test_predictions.csv", "text/csv", use_container_width=True)
    col1.download_button("Download Full Research Package", package, f"random_forest_research_package_{year}.zip", "application/zip", use_container_width=True)
    col2.download_button("Download Trained Model", model_bytes(st.session_state.model), "random_forest_model.joblib", "application/octet-stream", use_container_width=True)
    with st.expander("Additional CSV downloads"):
        a, b, c = st.columns(3)
        a.download_button("Engineered Dataset CSV", dataframe_csv(prepared.engineered_data), "engineered_dataset.csv", "text/csv", use_container_width=True)
        b.download_button("Metrics CSV", dataframe_csv(metrics_frame(results)), "classification_metrics.csv", "text/csv", use_container_width=True)
        c.download_button("Feature Importance CSV", dataframe_csv(results["feature_importance"]), "feature_importance.csv", "text/csv", use_container_width=True)

st.caption("Academic research utility only — results are not financial advice.")
