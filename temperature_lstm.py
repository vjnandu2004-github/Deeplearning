import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf


# ==============================
# 1. Load dataset
# ==============================

data = pd.read_csv(
    "temperature.csv"
)

print(data.head())

print("\nDataset shape:")
print(data.shape)


# ==============================
# 2. Select temperature column
# ==============================

temperatures = data[
    "Temperature"
].values


temperatures = temperatures.reshape(
    -1,
    1
)


# ==============================
# 3. Normalize data
# ==============================

scaler = MinMaxScaler()

scaled_data = scaler.fit_transform(
    temperatures
)


# ==============================
# 4. Create sequences
# ==============================

sequence_length = 10

X = []
y = []


for i in range(
    len(scaled_data) - sequence_length
):

    X.append(
        scaled_data[
            i:i + sequence_length
        ]
    )

    y.append(
        scaled_data[
            i + sequence_length
        ]
    )


X = np.array(X)
y = np.array(y)


print("\nX shape:", X.shape)
print("y shape:", y.shape)


# ==============================
# 5. Train-test split
# ==============================

split = int(
    len(X) * 0.8
)

X_train = X[:split]
X_test = X[split:]

y_train = y[:split]
y_test = y[split:]


print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==============================
# 6. Create LSTM model
# ==============================

model = tf.keras.Sequential([

    tf.keras.layers.LSTM(
        64,
        input_shape=(
            sequence_length,
            1
        )
    ),

    tf.keras.layers.Dense(32),

    tf.keras.layers.Dense(1)
])


# ==============================
# 7. Compile
# ==============================

model.compile(
    optimizer="adam",
    loss="mean_squared_error"
)


model.summary()


# ==============================
# 8. Train
# ==============================

history = model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)


# ==============================
# 9. Predict
# ==============================

predicted = model.predict(
    X_test
)


# ==============================
# 10. Convert back to temperature
# ==============================

predicted = scaler.inverse_transform(
    predicted
)

actual = scaler.inverse_transform(
    y_test
)


# ==============================
# 11. Plot actual vs predicted
# ==============================

plt.figure(figsize=(10, 5))

plt.plot(
    actual,
    label="Actual Temperature"
)

plt.plot(
    predicted,
    label="Predicted Temperature"
)

plt.xlabel("Time")
plt.ylabel("Temperature")

plt.title(
    "Actual vs Predicted Temperature"
)

plt.legend()

plt.show()


# ==============================
# 12. Plot training loss
# ==============================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title(
    "LSTM Training Loss"
)

plt.legend()

plt.show()
