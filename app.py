from fastapi import FastAPI, HTTPException
import yfinance as yf
from model import load_model, predict_prices

app = FastAPI(title="LSTM based Stock Prediction API")

lstm = None
W_out = None
b_out = None
min_val = None
max_val = None
model_error = None

try:
    lstm, W_out, b_out, min_val, max_val = load_model("lstm_stock_model.pkl")
    if lstm is None or W_out is None or b_out is None or min_val is None or max_val is None:
        raise ValueError("Model file must contain lstm_weights, W_out, b_out, min_val, and max_val.")
    print("Model weights loaded successfully into FastAPI backend!")
except FileNotFoundError:
    model_error = "lstm_stock_model.pkl not found. Prediction endpoint will return 503 until the model is available."
    print(f"Warning: {model_error}")
except Exception as e:
    model_error = f"Error loading model: {e}. Prediction endpoint will be unavailable."
    print(model_error)

@app.get("/predict")
def predict_stock(ticker: str = "RELIANCE.NS"):
    if model_error is not None or lstm is None or W_out is None or b_out is None or min_val is None or max_val is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Place lstm_stock_model.pkl in project root and restart the API.")

    try:
        try:
            data = yf.download(ticker, period="60d", progress=False)
        except Exception as e:
            raise HTTPException(status_code=404, detail=f"Error fetching ticker data: {e}")

        if data is None or data.empty:
            raise HTTPException(status_code=404, detail="Ticker data not found or no recent price data available.")

        raw_prices = data["Close"].values.astype(float).flatten()
        actual_prices, predicted_prices = predict_prices(
            lstm,
            W_out,
            b_out,
            min_val,
            max_val,
            raw_prices,
            context_length=10,
        )

        return {
            "ticker": ticker,
            "actual_prices": actual_prices,
            "predicted_prices": predicted_prices,
        }
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Internal server error during prediction. Check backend logs for details.",
        )
