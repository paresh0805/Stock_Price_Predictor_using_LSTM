# 📈 LSTM Stock Predictor — Live Insights

A lightweight machine learning application featuring a custom NumPy-based LSTM model and an interactive visualization frontend powered by **Streamlit**.

---

## 📸 Overview & Key Features

* **Custom NumPy LSTM:** Runs predictions without heavy frameworks like PyTorch or TensorFlow.
* **Direct Streamlit Deployment:** The dashboard loads the saved model directly for a single deployable app.
* **Optional FastAPI Backend:** `app.py` can still serve predictions through `/predict` when running locally.
* **Real-time Data:** Integrated with `yfinance` to automatically pull recent market data.

---

## 📁 Repository Structure

```text
.
├── app.py                  # Optional FastAPI server exposing the /predict endpoint
├── dashboard.py            # Streamlit dashboard UI
├── streamlit_app.py        # Streamlit entrypoint for deployment
├── model.py                # Shared model loading and prediction logic
├── lstm_stock_model.pkl    # (Required) Pickled model weights & scaling parameters
├── requirements.txt        # Python dependencies
├── .gitignore
└── README.md
```

> **⚠️ IMPORTANT:** This project requires a trained model pickle file (`lstm_stock_model.pkl`) in the root directory. If missing, the Streamlit app and the FastAPI backend cannot generate predictions.

---

## ⚙️ Model Requirements

The saved model pickle file should contain the standard LSTM weights and output parameters:

```python
{
    "lstm_weights": ...,
    "W_out": ...,
    "b_out": ...,
    "min_val": ...,
    "max_val": ...
}
```

The repo also supports legacy pickles that contain a raw `LSTMCell` instance, as long as the class is available at load time.

---

## 🚀 Quick Start

### 1. Environment Setup

Create and activate a virtual environment:

**Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

### 2. Running the Streamlit App Locally

```bash
streamlit run streamlit_app.py
```

### 3. Running the FastAPI Backend Locally (Optional)

If you want to use the optional API backend instead of the direct Streamlit model integration:

```bash
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Then run the Streamlit dashboard from `dashboard.py` in another terminal:

```bash
streamlit run dashboard.py
```

### 4. Usage

1. Open your browser to the Streamlit app URL shown in the terminal.
2. Enter a stock ticker in the sidebar (for example, `RELIANCE.NS`, `AAPL`, or `TSLA`).
3. Click **Fetch Predictions** to generate the analysis.

---

## ☁️ Deploying to Streamlit Cloud

1. Push this repository to GitHub.
2. Create a new Streamlit Cloud app and connect the repository.
3. Set the app file to `streamlit_app.py` and deploy.

The Streamlit app is self-contained and does not require the FastAPI backend when deployed this way.

---

## 💡 Troubleshooting & Notes

* If the app reports a missing model, verify `lstm_stock_model.pkl` is present in the repository root.
* If `yfinance` fails to return data for a ticker, try a valid symbol such as `AAPL` or `RELIANCE.NS`.
* The Streamlit cache improves responsiveness for repeated queries within **120 seconds**.

---

## 📄 License

This project is provided as-is for educational and experimental purposes.
