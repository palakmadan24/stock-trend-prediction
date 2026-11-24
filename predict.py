import argparse
from pathlib import Path

import joblib
import pandas as pd
import yfinance as yf
from train import add_features

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model" / "stock_model.pkl"


def main():
    parser = argparse.ArgumentParser(description="Predict next trading-day stock trend")
    parser.add_argument("--ticker", default=None, help="Ticker used for training; optional")
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model not found. First run: python train.py")

    bundle = joblib.load(MODEL_PATH)
    ticker = args.ticker or bundle["ticker"]

    if ticker != bundle["ticker"]:
        print("WARNING: This model was trained for", bundle["ticker"],
              "but you requested", ticker)
        print("For reliable use, train a separate model for this ticker first.")

    data = yf.download(
        ticker, period="6mo", interval="1d",
        auto_adjust=True, progress=False
    )
    if data.empty:
        raise RuntimeError("Could not download recent data. Check ticker/internet.")
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    featured = add_features(data).dropna()
    if featured.empty:
        raise RuntimeError("Not enough data to calculate technical indicators.")

    latest = featured.iloc[[-1]][bundle["features"]]
    model = bundle["model"]
    prediction = int(model.predict(latest)[0])
    confidence = None
    if hasattr(model, "predict_proba"):
        confidence = float(model.predict_proba(latest)[0][prediction])

    print("\n===== STOCK TREND PREDICTION =====")
    print("Stock ticker:", ticker)
    print("Model used:", bundle["model_name"])
    print("Latest trading date:", featured.index[-1].date())
    print("Latest closing price:", round(float(featured["Close"].iloc[-1]), 2))
    print("Predicted next trading-day trend:", "UP" if prediction == 1 else "DOWN")
    if confidence is not None:
        print(f"Model probability for predicted class: {confidence:.2%}")
    print("\nThis is an educational estimate, not financial advice.")


if __name__ == "__main__":
    main()
