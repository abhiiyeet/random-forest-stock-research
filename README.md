# Random Forest Stock Prediction Research Tool

A deliberately simple academic application for training a Random Forest classifier to predict whether the next trading day's closing price will be higher than today's close.

## Installation

```bash
python -m venv .venv
```

Activate the environment:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

Install and run:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## CSV format

Upload daily observations containing these columns:

```text
Date,Open,High,Low,Close,Volume
```

Common column-name variations such as `Closing Price` and `Total Traded Quantity` are recognized automatically. Numbers may contain commas.

With one CSV, the latest calendar year is automatically held out for testing and every earlier year is used for training. No random split or shuffling is used. You may instead upload a training CSV plus a chronologically later test CSV; December training history is then retained as rolling-feature context for January test observations without training on test labels.

Select one or more of the five predefined features, click **Train Model**, then click **Predict & Compare**. The tool reports true held-out classification results, a naive benchmark, and a simplified long-or-cash backtest. Exports include CSV files, a formatted Excel report, the trained joblib model, and a complete ZIP research package.

The prediction definition is:

- `UP / 1`: next day's close is greater than today's close.
- `DOWN / 0`: next day's close is less than or equal to today's close.

The final observation has no known outcome. It may receive a clearly marked unverified prediction, but it is never included in evaluation metrics.

This is an academic research tool, not a production trading platform. Its results are not financial advice.
