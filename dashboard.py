import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import re
import yfinance as yf
from datetime import datetime
from model import load_model, predict_prices

st.set_page_config(page_title="LSTM Insights", page_icon="📊", layout="wide")

st.title("LSTM Stock Predictor — Live Insights")
st.markdown(
    "A lightweight Streamlit application that loads the saved LSTM model directly and visualizes recent stock predictions."
    " Use the sidebar to configure ticker, display window, and prediction settings."
)

# --- SIDEBAR CONFIGURATION ---
st.sidebar.header("Configuration")

ticker_input = st.sidebar.text_input("NSE Stock Ticker (Yahoo Finance format):", value="RELIANCE.NS")

display_window = st.sidebar.selectbox("Display window (most recent days)", options=[30, 45, 60, 90], index=1)
context_length = st.sidebar.number_input("LSTM context length (days)", min_value=1, max_value=60, value=10)

st.sidebar.markdown("---")
assume_nse = st.sidebar.checkbox("Assume NSE suffix (.NS)", value=True)
fetch_button_sidebar = st.sidebar.button("Fetch Predictions")

# --- MAIN PAGE QUICK ACTIONS ---
st.markdown("---")
col_left, col_right = st.columns([3, 1])
with col_left:
    st.subheader("Recent Analysis")
with col_right:
    if st.button("🔁 Refresh"):
        fetch_button_sidebar = True

st.markdown("---")

model_error = None
lstm = None
W_out = None
b_out = None
min_val = None
max_val = None

try:
    lstm, W_out, b_out, min_val, max_val = load_model("lstm_stock_model.pkl")
    if lstm is None or W_out is None or b_out is None or min_val is None or max_val is None:
        raise ValueError("Model file must contain lstm_weights, W_out, b_out, min_val, and max_val.")
except FileNotFoundError:
    model_error = "Could not find lstm_stock_model.pkl. Please add the model file to the project root."
except Exception as e:
    model_error = f"Model loading failed: {e}"

@st.cache_data(ttl=120)
def fetch_predictions(ticker: str, context_length: int):
    if model_error is not None:
        raise ValueError(model_error)

    try:
        data = yf.download(ticker, period="60d", progress=False)
    except Exception as e:
        raise ValueError(f"Error fetching ticker data: {e}")

    if data is None or data.empty:
        raise ValueError("Ticker data not found or no recent price data is available.")

    raw_prices = data["Close"].values.astype(float).flatten()
    actual_prices, predicted_prices = predict_prices(
        lstm,
        W_out,
        b_out,
        min_val,
        max_val,
        raw_prices,
        context_length=context_length,
    )

    return {
        "ticker": ticker,
        "actual_prices": actual_prices,
        "predicted_prices": predicted_prices,
    }


def normalize_ticker(ticker: str, assume_nse: bool = True):
    """Normalize and validate ticker string. Returns (normalized_ticker, error_message)."""
    t = ticker.strip().upper()
    if assume_nse and "." not in t:
        t = f"{t}.NS"

    if len(t) == 0 or len(t) > 20:
        return None, "Ticker is empty or too long (max 20 chars)."

    if not re.match(r'^[A-Z0-9\.\-]+$', t):
        return None, "Ticker contains invalid characters. Use letters, numbers, dot or hyphen."

    return t, None


def render_results(result):
    actual = result.get("actual_prices", [])
    predicted = result.get("predicted_prices", [])

    if not actual or not predicted:
        st.warning("The model returned empty series. Check the model and ticker data.")
        return

    length = min(len(actual), len(predicted), display_window)
    actual_series = pd.Series(actual[-length:], name="Actual")
    predicted_series = pd.Series(predicted[-length:], name="Predicted")
    df = pd.concat(
        [actual_series.reset_index(drop=True), predicted_series.reset_index(drop=True)], axis=1
    )

    latest_actual = actual_series.iloc[-1]
    latest_pred = predicted_series.iloc[-1]
    delta = latest_pred - latest_actual
    delta_pct = (delta / latest_actual) * 100 if latest_actual != 0 else 0

    m1, m2, m3 = st.columns(3)
    m1.metric("Latest Actual Price", f"₹{latest_actual:,.2f}")
    m2.metric("LSTM Predicted Price", f"₹{latest_pred:,.2f}", delta=f"₹{delta:,.2f}")
    m3.metric("Prediction vs Actual (%)", f"{delta_pct:.2f}%")

    st.line_chart(
        df.rename(columns={0: "Actual", 1: "Predicted"}).set_index(
            pd.RangeIndex(start=1, stop=length + 1)
        )
    )

    with st.expander("Detailed chart and options"):
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df.index, df.iloc[:, 0], label="Actual", color="#111111", linewidth=2)
        ax.plot(df.index, df.iloc[:, 1], label="Predicted", color="#d62728", linestyle="--", linewidth=2)
        ax.set_xlabel("Recent Days")
        ax.set_ylabel("Price (₹)")
        ax.set_title(f"{result['ticker']} — Last {length} days")
        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig)

    csv = df.to_csv(index=False)
    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name=f"{result['ticker']}_predictions.csv",
        mime="text/csv",
    )
    st.success(f"Predictions loaded — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def run_dashboard():
    if model_error is not None:
        st.error(model_error)
        return

    if fetch_button_sidebar:
        ticker_norm, validation_error = normalize_ticker(ticker_input, assume_nse)
        if validation_error:
            st.error(validation_error)
        else:
            with st.spinner(f"Fetching predictions for {ticker_norm}..."):
                try:
                    result = fetch_predictions(ticker_norm, context_length)
                    render_results(result)
                except ValueError as e:
                    st.error(str(e))
                except Exception as e:
                    st.error(f"Unexpected error: {e}")
    else:
        st.info("Configure the ticker and press 'Fetch Predictions' in the sidebar to begin.")


if __name__ == "__main__":
    run_dashboard()
