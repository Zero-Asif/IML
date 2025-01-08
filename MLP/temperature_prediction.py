import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Simulated temperature data (replace with real data)
np.random.seed(42)
days = np.arange(0, 100)
temperature_data = np.sin(days / 10) * 10 + 20 + np.random.normal(0, 1, len(days))


window_size = 5  
X = []
y = []

for i in range(len(temperature_data) - window_size):
    X.append(temperature_data[i:i + window_size])
    y.append(temperature_data[i + window_size])

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the data
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(window_size,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)  
])


model.compile(optimizer='adam', loss='mse', metrics=['mae'])

history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=8, verbose=1)

loss, mae = model.evaluate(X_test, y_test)
print(f"Mean Absolute Error: {mae}")

predictions = model.predict(X_test)

plt.figure(figsize=(10, 5))
plt.plot(y_test, label="Actual Temperatures")
plt.plot(predictions, label="Predicted Temperatures", linestyle="--")
plt.legend()
plt.title("Temperature Prediction using MLP")
plt.xlabel("Sample Index")
plt.ylabel("Temperature")
plt.show()
