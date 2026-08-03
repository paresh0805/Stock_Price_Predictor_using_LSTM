import os
import pickle
import sys

import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)


def tanh(x):
    return np.tanh(x)


def tanh_derivative(x):
    return 1 - (np.tanh(x) ** 2)


class LSTMCell:
    def __init__(self, input_dim, hidden_dim):
        self.hidden_dim = hidden_dim
        self.concat_dim = input_dim + hidden_dim
        self.W_f = np.random.randn(hidden_dim, self.concat_dim) * 0.1
        self.b_f = np.zeros((hidden_dim, 1))
        self.W_i = np.random.randn(hidden_dim, self.concat_dim) * 0.1
        self.b_i = np.zeros((hidden_dim, 1))
        self.W_c = np.random.randn(hidden_dim, self.concat_dim) * 0.1
        self.b_c = np.zeros((hidden_dim, 1))
        self.W_o = np.random.randn(hidden_dim, self.concat_dim) * 0.1
        self.b_o = np.zeros((hidden_dim, 1))

    def forward(self, x_t, h_prev, C_prev):
        z = np.vstack((h_prev, x_t))
        f_t = sigmoid(np.dot(self.W_f, z) + self.b_f)
        i_t = sigmoid(np.dot(self.W_i, z) + self.b_i)
        c_tilde = tanh(np.dot(self.W_c, z) + self.b_c)
        o_t = sigmoid(np.dot(self.W_o, z) + self.b_o)

        C_t = (f_t * C_prev) + (i_t * c_tilde)
        h_t = o_t * tanh(C_t)
        return h_t, C_t, (x_t, h_prev, C_prev, z, f_t, i_t, c_tilde, o_t, C_t)

    def forward_sequence(self, X):
        self.caches = []
        h_t = np.zeros((self.hidden_dim, 1))
        C_t = np.zeros((self.hidden_dim, 1))
        hidden_states = []

        for i in range(X.shape[1]):
            h_t, C_t, cache = self.forward(X[:, i].reshape(-1, 1), h_t, C_t)
            hidden_states.append(h_t)
            self.caches.append(cache)

        return np.hstack(hidden_states)

    def backward_sequence(self, dh_layer):
        dW_f = np.zeros((self.hidden_dim, self.concat_dim))
        dW_i = np.zeros((self.hidden_dim, self.concat_dim))
        dW_c = np.zeros((self.hidden_dim, self.concat_dim))
        dW_o = np.zeros((self.hidden_dim, self.concat_dim))

        db_f = np.zeros((self.hidden_dim, 1))
        db_i = np.zeros((self.hidden_dim, 1))
        db_c = np.zeros((self.hidden_dim, 1))
        db_o = np.zeros((self.hidden_dim, 1))

        dh_next = np.zeros((self.hidden_dim, 1))
        dC_next = np.zeros((self.hidden_dim, 1))

        for t in reversed(range(len(self.caches))):
            x_t, h_prev, C_prev, z, f_t, i_t, c_tilde, o_t, C_t = self.caches[t]
            dh_from_layer = dh_layer[:, t].reshape(-1, 1)
            dh = dh_from_layer + dh_next
            do_t = dh * np.tanh(C_t)

            dC_spatial = dh * o_t * tanh_derivative(C_t)
            dC = dC_spatial + dC_next
            df_t = dC * C_prev
            di_t = dC * c_tilde
            dc_tilde = dC * i_t

            df_raw = df_t * sigmoid_derivative(f_t)
            di_raw = di_t * sigmoid_derivative(i_t)
            dc_raw = dc_tilde * tanh_derivative(c_tilde)
            do_raw = do_t * sigmoid_derivative(o_t)

            dW_f += np.dot(df_raw, z.T)
            db_f += df_raw
            dW_i += np.dot(di_raw, z.T)
            db_i += di_raw
            dW_c += np.dot(dc_raw, z.T)
            db_c += dc_raw
            dW_o += np.dot(do_raw, z.T)
            db_o += do_raw

            dC_next = dC * f_t
            dz = (
                np.dot(self.W_f.T, df_raw)
                + np.dot(self.W_i.T, di_raw)
                + np.dot(self.W_c.T, dc_raw)
                + np.dot(self.W_o.T, do_raw)
            )
            dh_next = dz[:self.hidden_dim, :]

        return {
            "dW_f": dW_f,
            "db_f": db_f,
            "dW_i": dW_i,
            "db_i": db_i,
            "dW_c": dW_c,
            "db_c": db_c,
            "dW_o": dW_o,
            "db_o": db_o,
        }

    def update_parameters(self, grads, learning_rate=0.01):
        self.W_f -= learning_rate * grads["dW_f"]
        self.b_f -= learning_rate * grads["db_f"]
        self.W_i -= learning_rate * grads["dW_i"]
        self.b_i -= learning_rate * grads["db_i"]
        self.W_c -= learning_rate * grads["dW_c"]
        self.b_c -= learning_rate * grads["db_c"]
        self.W_o -= learning_rate * grads["dW_o"]
        self.b_o -= learning_rate * grads["db_o"]


sys.modules["__main__"].LSTMCell = LSTMCell


def load_model(path="lstm_stock_model.pkl"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at {path}.")

    with open(path, "rb") as f:
        model_data = pickle.load(f)

    if isinstance(model_data, dict):
        return (
            model_data.get("lstm_weights"),
            model_data.get("W_out"),
            model_data.get("b_out"),
            model_data.get("min_val"),
            model_data.get("max_val"),
        )

    return model_data, None, None, None, None


def predict_prices(lstm, W_out, b_out, min_val, max_val, raw_prices, context_length=10):
    if lstm is None or W_out is None or b_out is None or min_val is None or max_val is None:
        raise ValueError("Model weights or scaling parameters are missing.")

    raw_prices = np.asarray(raw_prices, dtype=float).flatten()
    if raw_prices.size < context_length + 1:
        raise ValueError("Not enough price history to generate predictions.")

    if max_val == min_val:
        raise ValueError("Model min/max values are invalid (identical).")

    normalized_prices = (raw_prices - min_val) / (max_val - min_val)
    predictions = []
    actuals = []

    for i in range(len(normalized_prices) - context_length):
        x_seq = normalized_prices[i : i + context_length].reshape(1, -1)
        hidden_history = lstm.forward_sequence(x_seq)
        pred = np.dot(W_out, hidden_history) + b_out
        real_pred = pred[0, -1] * (max_val - min_val) + min_val
        predictions.append(float(real_pred))
        actuals.append(float(raw_prices[i + context_length]))

    return actuals, predictions