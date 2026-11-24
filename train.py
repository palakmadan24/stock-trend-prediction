import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "dataset"
MODEL_DIR = ROOT / "model"
GRAPH_DIR = ROOT / "graphs"
MODEL_PATH = MODEL_DIR / "stock_model.pkl"

FEATURES = [
    "Open", "High", "Low", "Close", "Volume",
    "SMA_5", "SMA_10", "EMA_10", "RSI_14",
    "MACD", "Daily_Return", "Volatility_5"
]


def add_features(data):
    """Create indicators using current/past data only."""
    df = data.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).title() for c in df.columns]

    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    close = df["Close"]
    df["SMA_5"] = close.rolling(5).mean()
    df["SMA_10"] = close.rolling(10).mean()
    df["EMA_10"] = close.ewm(span=10, adjust=False).mean()

    change = close.diff()
    average_gain = change.clip(lower=0).rolling(14).mean()
    average_loss = (-change.clip(upper=0)).rolling(14).mean()
    rs = average_gain / average_loss.replace(0, np.nan)
    df["RSI_14"] = 100 - (100 / (1 + rs))

    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    df["MACD"] = ema_12 - ema_26

    df["Daily_Return"] = close.pct_change()
    df["Volatility_5"] = df["Daily_Return"].rolling(5).std()
    return df


def download_stock_data(ticker, period):
    print(f"Downloading {ticker} historical data ({period})...")
    data = yf.download(
        ticker, period=period, interval="1d",
        auto_adjust=True, progress=False
    )
    if data.empty:
        raise RuntimeError("No data downloaded. Check ticker symbol and internet connection.")
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data.sort_index()


def build_models():
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced"))
        ]),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5, min_samples_leaf=10,
            class_weight="balanced", random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=5,
            class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", SVC(
                kernel="rbf", C=1.0, probability=True,
                class_weight="balanced", random_state=42
            ))
        ])
    }


def save_confusion_matrix(name, y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["DOWN", "UP"], yticklabels=["DOWN", "UP"]
    )
    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    safe_name = name.lower().replace(" ", "_")
    plt.savefig(GRAPH_DIR / f"confusion_matrix_{safe_name}.png", dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Stock Price Trend Prediction model training")
    parser.add_argument("--ticker", default="RELIANCE.NS", help="Yahoo Finance ticker, e.g. TCS.NS")
    parser.add_argument("--period", default="5y", help="Data period, e.g. 1y, 3y, 5y, 10y")
    args = parser.parse_args()

    for folder in (DATA_DIR, MODEL_DIR, GRAPH_DIR):
        folder.mkdir(exist_ok=True)

    raw = download_stock_data(args.ticker, args.period)
    raw.to_csv(DATA_DIR / "stock_data.csv")

    # Historical closing-price graph
    plt.figure(figsize=(12, 5))
    plt.plot(raw.index, raw["Close"], label="Closing Price")
    plt.title(f"{args.ticker} Historical Closing Price")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / "historical_stock_price.png", dpi=150)
    plt.close()

    data = add_features(raw)
    # Target at date t is whether close at t+1 is higher than close at t.
    data["Target"] = (data["Close"].shift(-1) > data["Close"]).astype(int)
    data = data.iloc[:-1].dropna(subset=FEATURES + ["Target"])

    if len(data) < 150:
        raise RuntimeError("Not enough clean data. Try a longer period such as 5y.")

    X = data[FEATURES]
    y = data["Target"].astype(int)

    # Chronological split: never shuffle time-series observations.
    split_index = int(len(data) * 0.8)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    models = build_models()
    results = []
    predictions = {}
    probabilities = {}
    fitted_models = {}

    print("\n========== MODEL EVALUATION ==========")
    print(f"Training rows: {len(X_train)} | Testing rows: {len(X_test)}")

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        prob = model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, pred)
        precision = precision_score(y_test, pred, zero_division=0)
        recall = recall_score(y_test, pred, zero_division=0)
        f1 = f1_score(y_test, pred, zero_division=0)
        auc = roc_auc_score(y_test, prob) if y_test.nunique() == 2 else np.nan

        results.append({
            "Algorithm": name, "Accuracy": accuracy,
            "Precision": precision, "Recall": recall,
            "F1-Score": f1, "ROC-AUC": auc
        })
        predictions[name] = pred
        probabilities[name] = prob
        fitted_models[name] = model

        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1-score : {f1:.4f}")
        print(f"ROC-AUC  : {auc:.4f}" if not np.isnan(auc) else "ROC-AUC  : unavailable")
        print("Confusion Matrix:\n", confusion_matrix(y_test, pred, labels=[0, 1]))
        print(classification_report(
            y_test, pred, labels=[0, 1],
            target_names=["DOWN", "UP"], zero_division=0
        ))
        save_confusion_matrix(name, y_test, pred)

    results_df = pd.DataFrame(results).sort_values(
        ["F1-Score", "Accuracy"], ascending=False
    )
    results_df.to_csv(GRAPH_DIR / "model_evaluation_metrics.csv", index=False)

    # Accuracy comparison
    plt.figure(figsize=(9, 5))
    sns.barplot(data=results_df, x="Algorithm", y="Accuracy")
    plt.ylim(0, 1)
    plt.title("Accuracy Comparison of ML Algorithms")
    plt.ylabel("Accuracy")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / "accuracy_comparison.png", dpi=150)
    plt.close()

    # ROC curve comparison
    plt.figure(figsize=(7, 6))
    if y_test.nunique() == 2:
        for name, prob in probabilities.items():
            fpr, tpr, _ = roc_curve(y_test, prob)
            auc = roc_auc_score(y_test, prob)
            plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
        plt.plot([0, 1], [0, 1], linestyle="--", label="Random baseline")
        plt.legend()
    plt.title("ROC Curve Comparison")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / "roc_curve_comparison.png", dpi=150)
    plt.close()

    # Actual vs predicted trend for the best model
    best_name = results_df.iloc[0]["Algorithm"]
    best_predictions = predictions[best_name]
    plot_count = min(150, len(y_test))
    plt.figure(figsize=(12, 5))
    plt.plot(y_test.index[-plot_count:], y_test.iloc[-plot_count:],
             label="Actual Trend", marker="o", markersize=3)
    plt.plot(y_test.index[-plot_count:], best_predictions[-plot_count:],
             label=f"Predicted Trend ({best_name})", marker="x", markersize=3)
    plt.yticks([0, 1], ["DOWN", "UP"])
    plt.title("Actual vs Predicted Stock Trend")
    plt.xlabel("Date")
    plt.ylabel("Trend")
    plt.legend()
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / "actual_vs_predicted_trend.png", dpi=150)
    plt.close()

    # Random Forest feature importance
    rf = fitted_models["Random Forest"]
    importance = pd.Series(
        rf.feature_importances_, index=FEATURES
    ).sort_values(ascending=False)
    plt.figure(figsize=(9, 6))
    sns.barplot(x=importance.values, y=importance.index)
    plt.title("Random Forest Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / "random_forest_feature_importance.png", dpi=150)
    plt.close()

    # Save the model selected by test F1-score, then accuracy as tie-breaker.
    best_model = fitted_models[best_name]
    bundle = {
        "model": best_model,
        "features": FEATURES,
        "ticker": args.ticker,
        "model_name": best_name,
        "target_definition": "1=next close higher; 0=next close lower or equal"
    }
    joblib.dump(bundle, MODEL_PATH)

    print("\n========== FINAL RESULT ==========")
    print(results_df.to_string(index=False))
    print(f"\nSelected model: {best_name}")
    print(f"Saved model: {MODEL_PATH}")
    print(f"Graphs and metrics saved in: {GRAPH_DIR}")
    print("Prediction meaning: 1 = UP, 0 = DOWN (or unchanged).")


if __name__ == "__main__":
    main()
