# Stock Price Trend Prediction Using Machine Learning

## Project objective
Use historical daily stock data and supervised machine learning to classify whether the next trading day's closing price will be higher than the current day's close.

- `1 = UP`: next trading day's close is higher.
- `0 = DOWN`: next trading day's close is lower than or equal to the current close.

This project predicts the direction, not the exact future stock price.

## Algorithms
1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier
4. Support Vector Machine (SVM)

The program evaluates all four and selects the model with the highest test F1-score (accuracy is used as a tie-breaker).

## Project files
- `app.py` - Streamlit web dashboard (this is what gets deployed)
- `train.py` - trains and evaluates the four models, saves graphs and `model/stock_model.pkl`
- `predict.py` - command-line next-day prediction using the saved model
- `database.py` - SQLite helper for prediction history
- `model/`, `dataset/`, `graphs/` - trained model, stock data and evaluation graphs
- `render.yaml` - Render deployment settings

## Requirements
- Python 3.12 (3.11 or newer works)
- Internet connection for Yahoo Finance data

## Install (Windows)
Open VS Code terminal in this folder:

```bash
py -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Run the web app locally
```bash
streamlit run app.py
```
Then open http://localhost:8501.

## Train and evaluate
Default stock: Reliance Industries (`RELIANCE.NS`), last five years:

```bash
python train.py
```

Other examples:

```bash
python train.py --ticker TCS.NS --period 5y
python train.py --ticker INFY.NS --period 3y
python train.py --ticker AAPL --period 5y
```

The training script:
- downloads daily stock data,
- saves it to `dataset/stock_data.csv`,
- creates SMA, EMA, RSI, MACD, daily return and volatility features,
- uses a chronological 80% training / 20% testing split,
- trains and evaluates all four algorithms,
- saves evaluation metrics and graphs in `graphs/`,
- saves the selected trained model to `model/stock_model.pkl`.

## Predict
After training:

```bash
python predict.py
```

The prediction script fetches recent data and predicts the next trading-day direction using the saved model.

## Evaluation outputs
- Accuracy, Precision, Recall, F1-score and ROC-AUC
- Confusion matrix for each algorithm
- Accuracy comparison graph
- ROC curve comparison
- Actual vs predicted trend graph
- Random Forest feature importance graph

## Important methodology note
The features for a date use only information available up to that date. The target is the following trading day's direction. Data is split chronologically, without shuffling, to reduce future-data leakage.

## Deploy on Render (via GitHub)

### 1. Push the project to GitHub
From inside the project folder:

```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

Make sure `app.py`, `requirements.txt`, `render.yaml`, `model/`, `dataset/` and `graphs/` are in the repository.

### 2. Create the service on Render
**Option A - Blueprint (uses `render.yaml`)**
1. Log in at https://render.com and click **New > Blueprint**.
2. Connect your GitHub account and select this repository.
3. Render reads `render.yaml`; click **Apply**.

**Option B - Manual Web Service**
1. Click **New > Web Service** and select this repository.
2. Set the values below, then click **Create Web Service**.

| Setting | Value |
|---|---|
| Runtime | Python 3 |
| Build Command | `pip install -r requirements.txt` |
| Start Command | `streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true` |
| Health Check Path | `/_stcore/health` |
| Environment variable | `PYTHON_VERSION` = `3.12.3` |

### 3. Open the app
The first build takes a few minutes. When it finishes, Render shows your public URL (`https://<service-name>.onrender.com`). Every `git push` to `main` redeploys automatically.

### Notes
- Free Render services sleep after about 15 minutes of inactivity, so the first visit afterwards can take around a minute to wake up.
- The SQLite prediction history (`prediction_history.db`) is created automatically on first run. On the free plan the disk is not persistent, so history resets on every redeploy or restart. Attach a Render Disk (paid) if you need it to persist.
- `scikit-learn` is pinned to `1.9.1` to match the version used to create `model/stock_model.pkl`. If you retrain with a different version, update `requirements.txt` to match.
- `joblib` and `seaborn` are used by `train.py` and `predict.py`, not by the web app.

## Limitations
Stock markets are noisy and affected by events that historical prices may not capture. Accuracy is not guaranteed. This is an academic project, not investment advice.
