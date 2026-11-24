
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import yfinance as yf
import sqlite3
import matplotlib.pyplot as plt

from pathlib import Path
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# =========================================================
# 1. CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="StockVision",
    page_icon="📈",
    layout="wide"
)

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "prediction_history.db"

STOCKS = {
    "Reliance Industries": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "Infosys": "INFY.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "State Bank of India": "SBIN.NS"
}

TARGET_THRESHOLD = 0.001

FEATURES = [
    "Return_1",
    "Return_3",
    "Return_5",
    "Return_10",
    "SMA_5_Ratio",
    "SMA_10_Ratio",
    "SMA_20_Ratio",
    "EMA_12_Ratio",
    "EMA_26_Ratio",
    "RSI_14",
    "MACD_Ratio",
    "MACD_Signal_Ratio",
    "BB_Position",
    "BB_Width",
    "Volatility_5",
    "Volatility_10",
    "Volume_Change",
    "Volume_Ratio",
    "Daily_Range",
    "Close_Position"
]

# =========================================================
# 2. THEME
# =========================================================

if "theme" not in st.session_state:
    st.session_state.theme = "Light"

if "page_override" not in st.session_state:
    st.session_state.page_override = None


def apply_theme():
    if st.session_state.theme == "Dark":
        bg = "#101827"
        card = "#1b2738"
        text = "#f1f5f9"
        border = "#344256"
    else:
        bg = "#f5f7fb"
        card = "#ffffff"
        text = "#172033"
        border = "#e2e8f0"

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: {bg};
            color: {text};
        }}

        [data-testid="stSidebar"] {{
            background-color: {card};
        }}

        [data-testid="stMetric"] {{
            background-color: {card};
            border: 1px solid {border};
            padding: 15px;
            border-radius: 12px;
        }}

        .main-heading {{
            font-size: 32px;
            font-weight: 800;
            color: {text};
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


apply_theme()

# =========================================================
# 3. HEADER AND THREE-DOT MENU
# =========================================================

left, right = st.columns([8, 1])

with left:
    st.markdown(
        '<div class="main-heading">📈 StockVision</div>',
        unsafe_allow_html=True
    )
    st.caption("ML Based Stock Price Trend Prediction")

with right:
    with st.popover("⋮"):
        st.markdown("### Settings")

        theme_choice = st.radio(
            "Appearance",
            ["Light", "Dark"],
            index=0 if st.session_state.theme == "Light" else 1
        )

        if theme_choice != st.session_state.theme:
            st.session_state.theme = theme_choice
            st.rerun()

        st.divider()

        if st.button("🎥 Screen Recording", use_container_width=True):
            st.session_state.page_override = "Screen Recording"
            st.rerun()

st.divider()

# =========================================================
# 4. DATABASE
# =========================================================

def create_database():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                ticker TEXT,
                prediction_date TEXT,
                closing_price REAL,
                trend TEXT,
                confidence REAL,
                prediction_type TEXT,
                forecast_price REAL
            )
        """)

        existing = [
            row[1]
            for row in conn.execute(
                "PRAGMA table_info(predictions)"
            ).fetchall()
        ]

        required = {
            "timestamp": "TEXT",
            "ticker": "TEXT",
            "prediction_date": "TEXT",
            "closing_price": "REAL",
            "trend": "TEXT",
            "confidence": "REAL",
            "prediction_type": "TEXT",
            "forecast_price": "REAL"
        }

        for column, data_type in required.items():
            if column not in existing:
                conn.execute(
                    f"ALTER TABLE predictions ADD COLUMN {column} {data_type}"
                )

        conn.commit()


def save_prediction(
    ticker,
    prediction_date,
    closing_price,
    trend,
    confidence,
    prediction_type,
    forecast_price=None
):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            INSERT INTO predictions (
                timestamp, ticker, prediction_date,
                closing_price, trend, confidence,
                prediction_type, forecast_price
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            ticker,
            str(prediction_date),
            float(closing_price),
            str(trend),
            float(confidence),
            str(prediction_type),
            forecast_price
        ))
        conn.commit()


def load_history():
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(
            "SELECT * FROM predictions ORDER BY id DESC",
            conn
        )


create_database()

# =========================================================
# 5. FETCH STOCK DATA
# =========================================================

@st.cache_data(ttl=3600)
def get_stock_data(ticker):
    try:
        data = yf.download(
            ticker,
            period="5y",
            interval="1d",
            auto_adjust=False,
            progress=False
        )

        if data is None or data.empty:
            return pd.DataFrame()

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        data.columns = [
            str(col).strip().title()
            for col in data.columns
        ]

        required = ["Open", "High", "Low", "Close", "Volume"]

        if not all(col in data.columns for col in required):
            return pd.DataFrame()

        data = data[required].copy()
        data = data.apply(pd.to_numeric, errors="coerce")
        data = data.dropna()
        data.index = pd.to_datetime(data.index)

        return data

    except Exception as error:
        print("Data error:", error)
        return pd.DataFrame()


# =========================================================
# 6. TECHNICAL INDICATORS AND TARGET
# =========================================================

def build_features(raw_data):
    df = raw_data.copy()

    close = df["Close"].astype(float)
    volume = df["Volume"].astype(float)

    # Returns
    df["Return_1"] = close.pct_change(1)
    df["Return_3"] = close.pct_change(3)
    df["Return_5"] = close.pct_change(5)
    df["Return_10"] = close.pct_change(10)

    # Moving averages
    sma_5 = close.rolling(5).mean()
    sma_10 = close.rolling(10).mean()
    sma_20 = close.rolling(20).mean()

    df["SMA_5_Ratio"] = close / sma_5 - 1
    df["SMA_10_Ratio"] = close / sma_10 - 1
    df["SMA_20_Ratio"] = close / sma_20 - 1

    # Exponential moving averages
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()

    df["EMA_12_Ratio"] = close / ema_12 - 1
    df["EMA_26_Ratio"] = close / ema_26 - 1

    # RSI
    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI_14"] = 100 - (100 / (1 + rs))
    df["RSI_14"] = df["RSI_14"].fillna(50)

    # MACD
    macd = ema_12 - ema_26
    signal = macd.ewm(span=9, adjust=False).mean()

    df["MACD_Ratio"] = macd / close
    df["MACD_Signal_Ratio"] = (macd - signal) / close

    # Bollinger Bands
    middle = close.rolling(20).mean()
    std = close.rolling(20).std()

    upper = middle + (2 * std)
    lower = middle - (2 * std)

    band_width = (upper - lower) / middle
    df["BB_Width"] = band_width
    df["BB_Position"] = (
        (close - lower) / (upper - lower).replace(0, np.nan)
    )

    # Volatility
    daily_returns = close.pct_change()

    df["Volatility_5"] = daily_returns.rolling(5).std()
    df["Volatility_10"] = daily_returns.rolling(10).std()

    # Volume indicators
    df["Volume_Change"] = volume.pct_change()

    volume_average = volume.rolling(20).mean()
    df["Volume_Ratio"] = volume / volume_average

    # Daily price range
    df["Daily_Range"] = (
        (df["High"] - df["Low"]) / close.replace(0, np.nan)
    )

    # Closing position within daily range
    daily_range = (df["High"] - df["Low"]).replace(0, np.nan)

    df["Close_Position"] = (
        (close - df["Low"]) / daily_range
    )

    # Next trading day's return and closing price
    df["Target_Return"] = close.shift(-1) / close - 1
    df["Target_Close"] = close.shift(-1)

    # 1 = UP, 0 = DOWN, NaN = neutral movement
    df["Target_Trend"] = np.select(
        [
            df["Target_Return"] > TARGET_THRESHOLD,
            df["Target_Return"] < -TARGET_THRESHOLD
        ],
        [1, 0],
        default=np.nan
    )

    df = df.replace([np.inf, -np.inf], np.nan)

    return df


# =========================================================
# 7. MODEL TRAINING AND COMPARISON
# =========================================================

@st.cache_resource
def train_models(ticker):
    raw_data = get_stock_data(ticker)

    if raw_data.empty or len(raw_data) < 150:
        return None

    featured = build_features(raw_data)

    # Keep rows having valid features and a non-neutral target
    model_data = featured.dropna(
        subset=FEATURES + ["Target_Trend", "Target_Close"]
    ).copy()

    if len(model_data) < 100:
        return None

    X = model_data[FEATURES]
    y = model_data["Target_Trend"].astype(int)

    # Chronological 70% train, 15% validation, 15% test
    n = len(model_data)

    train_end = int(n * 0.70)
    validation_end = int(n * 0.85)

    X_train = X.iloc[:train_end]
    y_train = y.iloc[:train_end]

    X_validation = X.iloc[train_end:validation_end]
    y_validation = y.iloc[train_end:validation_end]

    X_test = X.iloc[validation_end:]
    y_test = y.iloc[validation_end:]

    if (
        y_train.nunique() < 2
        or y_validation.empty
        or y_test.empty
    ):
        return None

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                C=0.5,
                random_state=42
            ))
        ]),

        "Random Forest": RandomForestClassifier(
            n_estimators=400,
            max_depth=6,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1
        ),

        "Extra Trees": ExtraTreesClassifier(
            n_estimators=400,
            max_depth=6,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
    }

    comparison_rows = []
    best_name = None
    best_score = -1
    best_model = None

    # Select model using validation data only
    for name, candidate in models.items():
        try:
            candidate.fit(X_train, y_train)
            validation_predictions = candidate.predict(X_validation)

            score = balanced_accuracy_score(
                y_validation,
                validation_predictions
            )

            comparison_rows.append({
                "Model": name,
                "Validation Accuracy": accuracy_score(
                    y_validation, validation_predictions
                ),
                "Validation Balanced Accuracy": score,
                "Validation F1": f1_score(
                    y_validation,
                    validation_predictions,
                    zero_division=0
                )
            })

            if score > best_score:
                best_score = score
                best_name = name
                best_model = candidate

        except Exception as error:
            print(f"{name} training error:", error)

    if best_model is None:
        return None

    # Refit selected model on train + validation
    X_train_validation = X.iloc[:validation_end]
    y_train_validation = y.iloc[:validation_end]

    best_model.fit(
        X_train_validation,
        y_train_validation
    )

    # Final evaluation on untouched test data
    test_predictions = best_model.predict(X_test)

    test_accuracy = accuracy_score(y_test, test_predictions)
    test_balanced_accuracy = balanced_accuracy_score(
        y_test, test_predictions
    )
    test_precision = precision_score(
        y_test, test_predictions, zero_division=0
    )
    test_recall = recall_score(
        y_test, test_predictions, zero_division=0
    )
    test_f1 = f1_score(
        y_test, test_predictions, zero_division=0
    )

    # Majority class baseline on the same test period
    majority_class = y_train_validation.mode().iloc[0]

    baseline_predictions = np.full(
        len(y_test),
        majority_class
    )

    baseline_accuracy = accuracy_score(
        y_test,
        baseline_predictions
    )

    baseline_balanced_accuracy = balanced_accuracy_score(
        y_test,
        baseline_predictions
    )

    # Fit a separate price regressor for future estimates
    regression_data = featured.dropna(
        subset=FEATURES + ["Target_Close"]
    )

    regressor = RandomForestRegressor(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )

    regressor.fit(
        regression_data[FEATURES],
        regression_data["Target_Close"]
    )

    # Latest feature row is used for next-trading-day prediction
    latest_feature_data = featured.dropna(
        subset=FEATURES
    )

    if latest_feature_data.empty:
        return None

    return {
        "raw_data": raw_data,
        "featured": featured,
        "model_data": model_data,
        "model": best_model,
        "model_name": best_name,
        "regressor": regressor,
        "comparison": pd.DataFrame(comparison_rows),
        "X_test": X_test,
        "y_test": y_test,
        "test_predictions": test_predictions,
        "accuracy": test_accuracy,
        "balanced_accuracy": test_balanced_accuracy,
        "precision": test_precision,
        "recall": test_recall,
        "f1": test_f1,
        "baseline_accuracy": baseline_accuracy,
        "baseline_balanced_accuracy": baseline_balanced_accuracy,
        "latest_features": latest_feature_data
    }


# =========================================================
# 8. HISTORICAL DATE PREDICTION
# =========================================================

def historical_prediction(bundle, selected_date):
    featured = bundle["featured"]

    selected_timestamp = pd.Timestamp(selected_date)

    if selected_timestamp not in featured.index:
        return None

    row = featured.loc[[selected_timestamp]]

    if row[FEATURES].isna().any(axis=None):
        return None

    model = bundle["model"]

    prediction = int(model.predict(row[FEATURES])[0])

    probabilities = model.predict_proba(row[FEATURES])[0]
    class_index = list(model.classes_).index(prediction)

    confidence = float(probabilities[class_index]) * 100

    return {
        "trend": "UP" if prediction == 1 else "DOWN",
        "confidence": confidence,
        "date": selected_timestamp,
        "open": float(row["Open"].iloc[0]),
        "high": float(row["High"].iloc[0]),
        "low": float(row["Low"].iloc[0]),
        "close": float(row["Close"].iloc[0])
    }


# =========================================================
# 9. FUTURE FORECAST
# =========================================================

def future_forecast(bundle, days):
    temporary_data = bundle["raw_data"].copy()
    regressor = bundle["regressor"]

    forecast_rows = []

    for step in range(1, days + 1):
        current_features = build_features(temporary_data)
        current_features = current_features.dropna(
            subset=FEATURES
        )

        if current_features.empty:
            break

        input_row = current_features[FEATURES].iloc[[-1]]

        previous_close = float(
            temporary_data["Close"].iloc[-1]
        )

        predicted_price = float(
            regressor.predict(input_row)[0]
        )

        predicted_price = max(predicted_price, 0.01)

        change = (
            (predicted_price - previous_close)
            / previous_close
        ) * 100

        trend = (
            "UP" if predicted_price >= previous_close
            else "DOWN"
        )

        next_date = (
            pd.Timestamp(temporary_data.index[-1])
            + pd.offsets.BDay(1)
        )

        forecast_rows.append({
            "Date": next_date.date(),
            "Predicted Price": round(predicted_price, 2),
            "Previous Price": round(previous_close, 2),
            "Change (%)": round(change, 2),
            "Trend": trend
        })

        new_row = pd.DataFrame(
            {
                "Open": [previous_close],
                "High": [max(previous_close, predicted_price)],
                "Low": [min(previous_close, predicted_price)],
                "Close": [predicted_price],
                "Volume": [
                    float(temporary_data["Volume"].iloc[-1])
                ]
            },
            index=[next_date]
        )

        temporary_data = pd.concat(
            [temporary_data, new_row]
        )

    return pd.DataFrame(forecast_rows)


# =========================================================
# 10. NAVIGATION
# =========================================================

pages = [
    "Dashboard",
    "Trend Prediction",
    "Future Forecast",
    "Prediction History",
    "Model Evaluation"
]

if st.session_state.page_override == "Screen Recording":
    page = "Screen Recording"
    st.session_state.page_override = None
else:
    page = st.sidebar.radio("Navigation", pages)

st.sidebar.divider()

stock_name = st.sidebar.selectbox(
    "Select Stock",
    list(STOCKS.keys())
)

ticker = STOCKS[stock_name]

st.sidebar.caption(f"Ticker: {ticker}")

# =========================================================
# 11. LOAD DATA AND TRAIN MODEL
# =========================================================

with st.spinner(
    f"Loading {stock_name} data and comparing ML models..."
):
    stock_data = get_stock_data(ticker)
    bundle = train_models(ticker)

if stock_data.empty or bundle is None:
    st.error(
        "Not enough valid stock data to train the models. "
        "Please check your internet connection or try again later."
    )
    st.stop()

latest_date = stock_data.index[-1]
latest_close = float(stock_data["Close"].iloc[-1])
previous_close = float(stock_data["Close"].iloc[-2])

daily_change = (
    (latest_close - previous_close) / previous_close
) * 100

# =========================================================
# 12. DASHBOARD
# =========================================================

if page == "Dashboard":

    st.title("📊 Stock Market Dashboard")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Latest Close",
        f"₹{latest_close:,.2f}",
        f"{daily_change:.2f}%"
    )

    c2.metric(
        "Day High",
        f"₹{stock_data['High'].iloc[-1]:,.2f}"
    )

    c3.metric(
        "Day Low",
        f"₹{stock_data['Low'].iloc[-1]:,.2f}"
    )

    c4.metric(
        "Volume",
        f"{stock_data['Volume'].iloc[-1]:,.0f}"
    )

    st.divider()

    st.subheader("Stock Closing Price")

    st.line_chart(
        stock_data[["Close"]].tail(180),
        use_container_width=True
    )

    left, right = st.columns(2)

    with left:
        st.subheader("Recent Market Data")
        st.dataframe(
            stock_data.tail(10).round(2),
            use_container_width=True
        )

    with right:
        st.subheader("Moving Averages")

        moving = stock_data[["Close"]].tail(180).copy()
        moving["SMA 5"] = moving["Close"].rolling(5).mean()
        moving["SMA 20"] = moving["Close"].rolling(20).mean()

        st.line_chart(
            moving,
            use_container_width=True
        )

    st.caption(
        f"Latest available market data: {latest_date.date()}"
    )

# =========================================================
# 13. TREND PREDICTION
# =========================================================

elif page == "Trend Prediction":

    st.title("🔮 Stock Trend Prediction")

    st.write(
        "Select a trading date to predict the following "
        "trading day's direction."
    )

    available_dates = [
        timestamp.date()
        for timestamp in bundle["featured"].dropna(
            subset=FEATURES
        ).index
    ]

    if available_dates:
        selected_date = st.date_input(
            "Select Trading Date",
            value=available_dates[-1],
            min_value=available_dates[30],
            max_value=available_dates[-1]
        )

        if st.button("Predict Trend", type="primary"):

            result = historical_prediction(
                bundle,
                selected_date
            )

            if result is None:
                st.warning(
                    "Could not generate prediction for this date."
                )
            else:
                st.subheader("Prediction Result")

                if result["trend"] == "UP":
                    st.success("Predicted Trend: UP 📈")
                else:
                    st.error("Predicted Trend: DOWN 📉")

                st.metric(
                    "Model Confidence",
                    f"{result['confidence']:.2f}%"
                )

                st.divider()
                st.subheader("Selected Date Market Details")

                a, b, c, d = st.columns(4)

                a.metric("Open", f"₹{result['open']:,.2f}")
                b.metric("High", f"₹{result['high']:,.2f}")
                c.metric("Low", f"₹{result['low']:,.2f}")
                d.metric("Close", f"₹{result['close']:,.2f}")

                st.caption(
                    "OHLC values are actual market data for the "
                    "selected date. The model predicts the next "
                    "trading day's direction."
                )

                save_prediction(
                    ticker,
                    selected_date,
                    result["close"],
                    result["trend"],
                    result["confidence"],
                    "Historical Trend"
                )

                st.success("Prediction saved successfully.")

# =========================================================
# 14. FUTURE FORECAST
# =========================================================

elif page == "Future Forecast":

    st.title("📅 Future Stock Price Forecast")

    forecast_days = st.selectbox(
        "Select Forecast Period",
        [5, 10, 20]
    )

    if st.button("Generate Forecast", type="primary"):

        with st.spinner("Generating forecast..."):
            forecast = future_forecast(
                bundle,
                forecast_days
            )

        if forecast.empty:
            st.warning("Forecast could not be generated.")
        else:
            st.dataframe(
                forecast,
                use_container_width=True,
                hide_index=True
            )

            st.subheader("Forecast Price Chart")

            st.line_chart(
                forecast.set_index("Date")[["Predicted Price"]],
                use_container_width=True
            )

            last_price = float(
                forecast["Predicted Price"].iloc[-1]
            )

            change = (
                (last_price - latest_close) / latest_close
            ) * 100

            x1, x2, x3 = st.columns(3)

            x1.metric(
                "Current Price",
                f"₹{latest_close:,.2f}"
            )

            x2.metric(
                "Last Forecast Price",
                f"₹{last_price:,.2f}"
            )

            x3.metric(
                "Estimated Change",
                f"{change:.2f}%"
            )

            for _, row in forecast.iterrows():
                save_prediction(
                    ticker,
                    row["Date"],
                    latest_close,
                    row["Trend"],
                    0,
                    "Future Forecast",
                    row["Predicted Price"]
                )

            csv_data = forecast.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                "⬇️ Download Forecast CSV",
                data=csv_data,
                file_name="stock_forecast.csv",
                mime="text/csv"
            )

            st.info(
                "These are model-based estimates, not guaranteed "
                "future prices or investment advice."
            )

# =========================================================
# 15. PREDICTION HISTORY
# =========================================================

elif page == "Prediction History":

    st.title("🗂️ Prediction History")

    history = load_history()

    if history.empty:
        st.info("No saved predictions available.")

    else:
        history = history[
            history["ticker"] == ticker
        ].copy()

        if history.empty:
            st.info("No history found for this stock.")
        else:
            st.metric(
                "Total Saved Predictions",
                len(history)
            )

            st.dataframe(
                history,
                use_container_width=True,
                hide_index=True
            )

            csv_data = history.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                "⬇️ Download History CSV",
                data=csv_data,
                file_name="stock_prediction_history.csv",
                mime="text/csv"
            )

# =========================================================
# 16. MODEL EVALUATION
# =========================================================

elif page == "Model Evaluation":

    st.title("🧪 Model Evaluation")

    st.markdown(
        f"**Selected model:** {bundle['model_name']}"
    )

    st.subheader("Final Test Results")

    a, b, c, d, e = st.columns(5)

    a.metric(
        "Accuracy",
        f"{bundle['accuracy'] * 100:.2f}%"
    )

    b.metric(
        "Balanced Accuracy",
        f"{bundle['balanced_accuracy'] * 100:.2f}%"
    )

    c.metric(
        "Precision",
        f"{bundle['precision'] * 100:.2f}%"
    )

    d.metric(
        "Recall",
        f"{bundle['recall'] * 100:.2f}%"
    )

    e.metric(
        "F1 Score",
        f"{bundle['f1'] * 100:.2f}%"
    )

    st.divider()

    st.subheader("Model Comparison on Validation Data")

    comparison = bundle["comparison"].copy()

    for column in [
        "Validation Accuracy",
        "Validation Balanced Accuracy",
        "Validation F1"
    ]:
        comparison[column] = (
            comparison[column] * 100
        ).round(2)

    st.dataframe(
        comparison,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader("Baseline Comparison")

    baseline_col1, baseline_col2 = st.columns(2)

    baseline_col1.metric(
        "Majority Baseline Accuracy",
        f"{bundle['baseline_accuracy'] * 100:.2f}%"
    )

    baseline_col2.metric(
        "Majority Baseline Balanced Accuracy",
        f"{bundle['baseline_balanced_accuracy'] * 100:.2f}%"
    )

    st.caption(
        "The baseline always predicts the most common class "
        "in the training and validation period."
    )

    st.divider()

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(
        bundle["y_test"],
        bundle["test_predictions"],
        labels=[0, 1]
    )

    fig, ax = plt.subplots(figsize=(6, 4))

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["DOWN", "UP"]
    )

    display.plot(
        ax=ax,
        cmap="Blues",
        colorbar=False
    )

    ax.set_title("Final Test Data")
    st.pyplot(fig)
    plt.close(fig)

    st.info(
        "The final test period was not used to choose the model. "
        "Accuracy is measured on unseen chronological data. "
        "A result below 50% can still occur because market "
        "movements are noisy and difficult to predict."
    )

# =========================================================
# 17. SCREEN RECORDING
# =========================================================

elif page == "Screen Recording":

    st.title("🎥 Screen Recording")

    st.write(
        "Use the buttons below to record your screen. "
        "Your browser may ask you to select a screen or window."
    )

    components.html(
        """
        <div style="
            font-family: Arial;
            padding: 18px;
            border-radius: 12px;
            background: #f1f5f9;
            color: #172033;
        ">
            <h3>Screen Recorder</h3>

            <button id="start" style="
                padding: 10px 18px;
                background: #16a34a;
                color: white;
                border: none;
                border-radius: 7px;
                cursor: pointer;
            ">Start Recording</button>

            <button id="stop" disabled style="
                padding: 10px 18px;
                background: #dc2626;
                color: white;
                border: none;
                border-radius: 7px;
                cursor: pointer;
            ">Stop Recording</button>

            <p id="status">Status: Not recording</p>
            <p id="message"></p>
        </div>

        <script>
        let recorder;
        let stream;
        let chunks = [];

        const start = document.getElementById("start");
        const stop = document.getElementById("stop");
        const status = document.getElementById("status");
        const message = document.getElementById("message");

        start.onclick = async () => {
            try {
                if (!navigator.mediaDevices ||
                    !navigator.mediaDevices.getDisplayMedia) {
                    message.innerText =
                        "Screen recording is not supported here.";
                    return;
                }

                stream = await navigator.mediaDevices.getDisplayMedia({
                    video: true,
                    audio: true
                });

                chunks = [];
                recorder = new MediaRecorder(stream);

                recorder.ondataavailable = event => {
                    if (event.data.size > 0) {
                        chunks.push(event.data);
                    }
                };

                recorder.onstop = () => {
                    const blob = new Blob(chunks, {
                        type: "video/webm"
                    });

                    const url = URL.createObjectURL(blob);
                    const link = document.createElement("a");

                    link.href = url;
                    link.download = "StockVision_Recording.webm";
                    link.click();

                    URL.revokeObjectURL(url);

                    stream.getTracks().forEach(
                        track => track.stop()
                    );

                    status.innerText = "Status: Recording saved";
                };

                recorder.start();

                start.disabled = true;
                stop.disabled = false;
                status.innerText = "Status: Recording...";

                stream.getVideoTracks()[0].addEventListener(
                    "ended",
                    () => {
                        if (recorder.state !== "inactive") {
                            recorder.stop();
                        }
                        start.disabled = false;
                        stop.disabled = true;
                    }
                );

            } catch (error) {
                message.innerText = error.message;
            }
        };

        stop.onclick = () => {
            if (recorder && recorder.state !== "inactive") {
                recorder.stop();
            }

            start.disabled = false;
            stop.disabled = true;
        };
        </script>
        """,
        height=240
    )

    st.warning(
        "Screen recording depends on browser permissions and "
        "deployment support. If capture is blocked in the embedded "
        "recorder, use your computer's screen recording tool."
    )

# =========================================================
# 18. FOOTER
# =========================================================

st.divider()

st.markdown(
    """
    <div style="text-align:center; opacity:0.7; padding:10px;">
        <b>StockVision</b> | Machine Learning Stock Trend Prediction
        <br>
        Built with Python, Streamlit, Scikit-learn, Yahoo Finance and SQLite
    </div>
    """,
    unsafe_allow_html=True
)
