"""Leakage-aware data and modelling pipeline for the research application."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, BinaryIO

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


REQUIRED_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]
FEATURE_LABELS = {
    "MA_5": "5-Day Moving Average",
    "MA_20": "20-Day Moving Average",
    "RSI_14": "RSI (14)",
    "Momentum_5": "5-Day Momentum",
    "Volume_Change": "Daily Volume Change",
}
ALIASES = {
    "Date": {"date", "datetime", "timestamp", "tradingdate", "tradedate"},
    "Open": {"open", "openprice", "openingprice"},
    "High": {"high", "highprice"},
    "Low": {"low", "lowprice"},
    "Close": {"close", "closeprice", "closingprice", "adjclose", "adjustedclose"},
    "Volume": {"volume", "totalvolume", "totaltradedquantity", "tradedvolume"},
}
DEFAULT_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "min_samples_split": 20,
    "min_samples_leaf": 10,
    "max_features": "sqrt",
    "random_state": 42,
    "n_jobs": -1,
}


class DataValidationError(ValueError):
    """A concise error safe to display in the interface."""


@dataclass
class PreparedData:
    raw_data: pd.DataFrame
    engineered_data: pd.DataFrame
    train_data: pd.DataFrame
    test_data: pd.DataFrame
    test_evaluation: pd.DataFrame
    latest_unverified: pd.DataFrame
    selected_features: list[str]
    raw_rows: int
    usable_rows: int
    training_period: str
    test_period: str
    separate_test: bool


def _key(value: object) -> str:
    return "".join(ch for ch in str(value).lower().strip() if ch.isalnum())


def _excel_sheet_data(sheet: pd.DataFrame, sheet_name: str) -> pd.DataFrame:
    """Find the OHLCV header row in a worksheet and return its data block."""
    required_aliases = [aliases for aliases in ALIASES.values()]
    for row_index in range(min(len(sheet), 25)):
        keys = {_key(value) for value in sheet.iloc[row_index].tolist()}
        if all(keys.intersection(aliases) for aliases in required_aliases):
            frame = sheet.iloc[row_index + 1 :].copy()
            frame.columns = sheet.iloc[row_index].tolist()
            return frame.dropna(how="all")
    raise DataValidationError(
        f'The worksheet "{sheet_name}" does not contain a recognizable OHLCV header row.'
    )


def clean_stock_data(source: BinaryIO | str | pd.DataFrame) -> pd.DataFrame:
    """Read, recognize and clean CSV/Excel OHLCV data without future information."""
    try:
        if isinstance(source, pd.DataFrame):
            frame = source.copy()
        else:
            name = str(getattr(source, "name", source)).lower()
            if name.endswith((".xlsx", ".xlsm", ".xls")):
                sheets = pd.read_excel(source, sheet_name=None, header=None)
                populated = [(sheet_name, sheet) for sheet_name, sheet in sheets.items() if not sheet.dropna(how="all").empty]
                if not populated:
                    raise DataValidationError("The uploaded workbook contains no data rows.")
                frame = pd.concat(
                    [_excel_sheet_data(sheet, sheet_name) for sheet_name, sheet in populated],
                    ignore_index=True,
                )
            else:
                frame = pd.read_csv(source)
    except DataValidationError:
        raise
    except Exception as exc:
        raise DataValidationError("The uploaded file could not be read as CSV or Excel data.") from exc
    if frame.empty:
        raise DataValidationError("The uploaded file contains no data rows.")

    normalized = {_key(c): c for c in frame.columns}
    rename: dict[str, str] = {}
    for standard, aliases in ALIASES.items():
        found = next((normalized[a] for a in aliases if a in normalized), None)
        if found is None:
            raise DataValidationError(
                f"The uploaded file does not contain a recognizable {standard} column."
            )
        rename[found] = standard
    frame = frame.rename(columns=rename)[REQUIRED_COLUMNS].copy()
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce", dayfirst=False)
    for column in REQUIRED_COLUMNS[1:]:
        values = frame[column].astype(str).str.replace(",", "", regex=False).str.strip()
        frame[column] = pd.to_numeric(values, errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan)
    frame = frame.dropna(subset=REQUIRED_COLUMNS)
    frame = frame.drop_duplicates(subset="Date", keep="last").sort_values("Date").reset_index(drop=True)
    invalid = (frame[["Open", "High", "Low", "Close"]] <= 0).any(axis=1) | (frame["Volume"] < 0)
    frame = frame.loc[~invalid].reset_index(drop=True)
    if frame.empty:
        raise DataValidationError("The uploaded file has no usable OHLCV observations after cleaning.")
    if len(frame) < 2:
        raise DataValidationError("At least two valid chronological observations are required.")
    return frame


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Construct features using only values at time t or earlier (never centered/future data)."""
    data = frame.sort_values("Date").copy()
    data["MA_5"] = data["Close"].rolling(5, min_periods=5).mean()
    data["MA_20"] = data["Close"].rolling(20, min_periods=20).mean()
    delta = data["Close"].diff()
    avg_gain = delta.clip(lower=0).rolling(14, min_periods=14).mean()
    avg_loss = (-delta.clip(upper=0)).rolling(14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    data["RSI_14"] = 100 - (100 / (1 + rs))
    data.loc[(avg_loss == 0) & (avg_gain > 0), "RSI_14"] = 100.0
    data.loc[(avg_loss == 0) & (avg_gain == 0), "RSI_14"] = 50.0
    data["Momentum_5"] = data["Close"] / data["Close"].shift(5) - 1
    prior_volume = data["Volume"].shift(1).replace(0, np.nan)
    data["Volume_Change"] = data["Volume"] / prior_volume - 1
    data["Daily_Return"] = data["Close"].pct_change()
    data = data.replace([np.inf, -np.inf], np.nan)
    return data


def _period(data: pd.DataFrame) -> str:
    start, end = data["Date"].min(), data["Date"].max()
    return str(start.year) if start.year == end.year else f"{start.year}–{end.year}"


def prepare_data(
    training_source: BinaryIO | str | pd.DataFrame,
    selected_features: list[str],
    test_source: BinaryIO | str | pd.DataFrame | None = None,
) -> PreparedData:
    if not selected_features:
        raise DataValidationError("At least one feature must be selected.")
    unknown = set(selected_features) - set(FEATURE_LABELS)
    if unknown:
        raise DataValidationError("An unsupported model feature was selected.")

    first = clean_stock_data(training_source)
    separate = test_source is not None
    if separate:
        second = clean_stock_data(test_source)
        if second["Date"].min() <= first["Date"].max():
            raise DataValidationError("The test dataset must occur after the training dataset.")
        first = first.assign(Dataset_Split="TRAIN")
        second = second.assign(Dataset_Split="TEST")
        raw = pd.concat([first, second], ignore_index=True).sort_values("Date").reset_index(drop=True)
    else:
        years = sorted(first["Date"].dt.year.unique())
        if len(years) < 2:
            raise DataValidationError("The dataset must contain observations from at least two calendar years.")
        test_year = years[-1]
        raw = first.assign(
            Dataset_Split=np.where(first["Date"].dt.year == test_year, "TEST", "TRAIN")
        )

    data = engineer_features(raw)
    data["Next_Close"] = data["Close"].shift(-1)
    next_split = data["Dataset_Split"].shift(-1)
    # A label is valid only when its next observation exists in the same split.
    # This prevents the first test price from becoming a training label.
    genuine_target = data["Next_Close"].notna() & data["Dataset_Split"].eq(next_split)
    data["Target"] = pd.Series(pd.NA, index=data.index, dtype="Int64")
    data.loc[genuine_target, "Target"] = (
        data.loc[genuine_target, "Next_Close"] > data.loc[genuine_target, "Close"]
    ).astype(int)
    data["Next_Day_Return"] = np.where(
        genuine_target, data["Next_Close"] / data["Close"] - 1, np.nan
    )

    feature_ready = data[selected_features].notna().all(axis=1)
    usable = data.loc[feature_ready].copy()
    train = usable.loc[(usable["Dataset_Split"] == "TRAIN") & usable["Target"].notna()].copy()
    test_all = usable.loc[usable["Dataset_Split"] == "TEST"].copy()
    test_eval = test_all.loc[test_all["Target"].notna()].copy()
    latest = test_all.loc[test_all["Target"].isna()].tail(1).copy()
    if train.empty:
        raise DataValidationError("Not enough historical observations are available for training with the selected features.")
    if test_eval.empty:
        raise DataValidationError("The test period has no observations with a known next-day outcome.")
    if train["Target"].nunique() < 2:
        raise DataValidationError("Training data must contain both UP and DOWN outcomes.")

    engineered_cols = REQUIRED_COLUMNS + list(FEATURE_LABELS) + ["Next_Close", "Target", "Dataset_Split"]
    return PreparedData(
        raw_data=raw[REQUIRED_COLUMNS].copy(),
        engineered_data=data[engineered_cols].copy(),
        train_data=train,
        test_data=test_all,
        test_evaluation=test_eval,
        latest_unverified=latest,
        selected_features=list(selected_features),
        raw_rows=len(raw),
        usable_rows=len(usable),
        training_period=_period(train),
        test_period=_period(test_all),
        separate_test=separate,
    )


def train_model(prepared: PreparedData, parameters: dict[str, Any] | None = None):
    params = {**DEFAULT_PARAMS, **(parameters or {})}
    model = RandomForestClassifier(**params)
    # Explicit allow-list: target, future close/return, and every other column are excluded.
    x_train = prepared.train_data.loc[:, prepared.selected_features]
    y_train = prepared.train_data["Target"].astype(int)
    model.fit(x_train, y_train)
    return model, params


def _probabilities(model, x: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    probabilities = model.predict_proba(x)
    by_class = {int(cls): probabilities[:, i] for i, cls in enumerate(model.classes_)}
    return by_class.get(0, np.zeros(len(x))), by_class.get(1, np.zeros(len(x)))


def classification_metrics(actual, predicted, probability_up=None) -> dict[str, float | None]:
    metrics: dict[str, float | None] = {
        "Accuracy": accuracy_score(actual, predicted),
        "Precision": precision_score(actual, predicted, zero_division=0),
        "Recall": recall_score(actual, predicted, zero_division=0),
        "F1": f1_score(actual, predicted, zero_division=0),
    }
    metrics["ROC-AUC"] = (
        roc_auc_score(actual, probability_up) if probability_up is not None and len(np.unique(actual)) == 2 else None
    )
    return metrics


def trading_metrics(returns: pd.Series) -> tuple[dict[str, float], pd.Series, pd.Series]:
    returns = pd.Series(returns, dtype=float).fillna(0.0)
    wealth = (1 + returns).cumprod()
    drawdown = wealth / wealth.cummax() - 1
    n = len(returns)
    cumulative = wealth.iloc[-1] - 1 if n else 0.0
    annualized = wealth.iloc[-1] ** (252 / n) - 1 if n and wealth.iloc[-1] > 0 else -1.0
    volatility = returns.std(ddof=1) * np.sqrt(252) if n > 1 else 0.0
    sharpe = returns.mean() / returns.std(ddof=1) * np.sqrt(252) if n > 1 and returns.std(ddof=1) > 0 else 0.0
    return {
        "Cumulative Return": cumulative,
        "Annualized Return": annualized,
        "Annualized Volatility": volatility,
        "Sharpe Ratio": sharpe,
        "Maximum Drawdown": drawdown.min() if n else 0.0,
    }, wealth, drawdown


def evaluate_model(model, prepared: PreparedData, transaction_cost_percent: float = 0.0) -> dict[str, Any]:
    test = prepared.test_evaluation.copy()
    x_test = test.loc[:, prepared.selected_features]
    prediction = model.predict(x_test).astype(int)
    probability_down, probability_up = _probabilities(model, x_test)
    actual = test["Target"].astype(int).to_numpy()

    predictions = test[REQUIRED_COLUMNS].copy()
    predictions["Actual_Next_Close"] = test["Next_Close"].to_numpy()
    predictions["Actual_Target"] = actual
    predictions["Actual_Direction"] = np.where(actual == 1, "UP", "DOWN")
    predictions["Prediction"] = prediction
    predictions["Predicted_Direction"] = np.where(prediction == 1, "UP", "DOWN")
    predictions["Probability_DOWN"] = probability_down
    predictions["Probability_UP"] = probability_up
    predictions["Correct"] = prediction == actual
    predictions["Next_Day_Return"] = test["Next_Day_Return"].to_numpy()

    cost = float(transaction_cost_percent) / 100.0
    positions = pd.Series(prediction, index=predictions.index, dtype=float)
    changes = positions.diff().abs()
    changes.iloc[0] = positions.iloc[0]  # charge only if the strategy initially enters long
    predictions["Strategy_Return"] = positions * predictions["Next_Day_Return"] - changes * cost
    predictions["Buy_Hold_Return"] = predictions["Next_Day_Return"]

    metrics = classification_metrics(actual, prediction, probability_up)
    baseline = (test["Daily_Return"].fillna(0).to_numpy() > 0).astype(int)
    baseline_metrics = classification_metrics(actual, baseline)
    comparison = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1"],
        "Random Forest": [metrics[k] for k in ["Accuracy", "Precision", "Recall", "F1"]],
        "Naive Baseline": [baseline_metrics[k] for k in ["Accuracy", "Precision", "Recall", "F1"]],
    })
    comparison["Difference"] = comparison["Random Forest"] - comparison["Naive Baseline"]

    rf_stats, rf_wealth, rf_drawdown = trading_metrics(predictions["Strategy_Return"])
    bh_stats, bh_wealth, bh_drawdown = trading_metrics(predictions["Buy_Hold_Return"])
    trading = pd.DataFrame({
        "Date": predictions["Date"].to_numpy(),
        "RF Daily Return": predictions["Strategy_Return"].to_numpy(),
        "Buy-and-Hold Daily Return": predictions["Buy_Hold_Return"].to_numpy(),
        "RF Cumulative Value": rf_wealth.to_numpy(),
        "Buy-and-Hold Cumulative Value": bh_wealth.to_numpy(),
        "RF Drawdown": rf_drawdown.to_numpy(),
        "Buy-and-Hold Drawdown": bh_drawdown.to_numpy(),
    })
    importance = pd.DataFrame({
        "Feature": prepared.selected_features,
        "Display Feature": [FEATURE_LABELS[f] for f in prepared.selected_features],
        "Importance": model.feature_importances_,
    }).sort_values("Importance", ascending=False).reset_index(drop=True)
    importance["Rank"] = np.arange(1, len(importance) + 1)

    latest_prediction = None
    if not prepared.latest_unverified.empty:
        row = prepared.latest_unverified.iloc[[-1]]
        pred = int(model.predict(row[prepared.selected_features])[0])
        _, up = _probabilities(model, row[prepared.selected_features])
        latest_prediction = {"Date": row["Date"].iloc[0], "Prediction": pred, "Probability_UP": float(up[0])}

    return {
        "predictions": predictions.reset_index(drop=True),
        "metrics": metrics,
        "confusion_matrix": confusion_matrix(actual, prediction, labels=[0, 1]),
        "baseline_metrics": baseline_metrics,
        "benchmark_comparison": comparison,
        "feature_importance": importance,
        "trading_performance": trading,
        "trading_metrics": {"RF Strategy": rf_stats, "Buy & Hold": bh_stats},
        "latest_prediction": latest_prediction,
    }
