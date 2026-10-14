import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, GRU, Dense

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
df = pd.read_csv(url)
series = df.iloc[:, 1].values.astype("float32").reshape(-1, 1)

print("Dataset shape:", series.shape)
print(df.head())

plt.figure(figsize=(10, 4))
plt.plot(series)
plt.title("International Airline Passengers (1949-1960)")
plt.xlabel("Month")
plt.ylabel("Passengers (thousands)")
plt.tight_layout()
plt.savefig("raw_series.png")
plt.show()

scaler = MinMaxScaler()
scaled = scaler.fit_transform(series)

look_back = 12
X, y = [], []
for i in range(len(scaled) - look_back):
    X.append(scaled[i:i + look_back])
    y.append(scaled[i + look_back])
X, y = np.array(X), np.array(y)

split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
print("Train shape:", X_train.shape, "Test shape:", X_test.shape)


def run(layer, name):
    model = Sequential([layer(50, activation="tanh", input_shape=(look_back, 1)), Dense(1)])
    model.compile(optimizer="adam", loss="mse")
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                        epochs=100, batch_size=8, verbose=0)

    train_pred = scaler.inverse_transform(model.predict(X_train, verbose=0))
    test_pred = scaler.inverse_transform(model.predict(X_test, verbose=0))
    y_train_inv = scaler.inverse_transform(y_train)
    y_test_inv = scaler.inverse_transform(y_test)

    train_rmse = np.sqrt(mean_squared_error(y_train_inv, train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test_inv, test_pred))
    print(f"[{name}] Train RMSE: {train_rmse:.2f} | Test RMSE: {test_rmse:.2f}")

    plt.figure(figsize=(6, 4))
    plt.plot(history.history["loss"], label="train loss")
    plt.plot(history.history["val_loss"], label="val loss")
    plt.title(f"{name} - Loss curve")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{name.lower()}_loss.png")
    plt.show()

    return test_pred, y_test_inv, test_rmse


rnn_pred, y_test_inv, rnn_rmse = run(SimpleRNN, "SimpleRNN")
lstm_pred, _, lstm_rmse = run(LSTM, "LSTM")
gru_pred, _, gru_rmse = run(GRU, "GRU")

plt.figure(figsize=(10, 5))
plt.plot(y_test_inv, label="Actual", linewidth=2)
plt.plot(rnn_pred, label=f"SimpleRNN (RMSE={rnn_rmse:.1f})")
plt.plot(lstm_pred, label=f"LSTM (RMSE={lstm_rmse:.1f})")
plt.plot(gru_pred, label=f"GRU (RMSE={gru_rmse:.1f})")
plt.title("Airline Passengers Forecast: RNN vs LSTM vs GRU")
plt.xlabel("Time step (test set)")
plt.ylabel("Passengers (thousands)")
plt.legend()
plt.tight_layout()
plt.savefig("rnn_vs_lstm_forecast.png")
plt.show()

print(f"Summary -> SimpleRNN: {rnn_rmse:.2f} | LSTM: {lstm_rmse:.2f} | GRU: {gru_rmse:.2f}")
