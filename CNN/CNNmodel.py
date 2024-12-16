import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Dropout
from tensorflow.keras.optimizers import Adam

data = pd.read_csv('https://raw.githubusercontent.com/Zero-Asif/data/refs/heads/main/electric-production.csv')

print(data.head())
print(data.info())

X = data.iloc[:, 1:].values  
y = data.iloc[:, 1].values   


X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print(f"Training data shape: {X_train.shape}, {y_train.shape}")
print(f"Validation data shape: {X_val.shape}, {y_val.shape}")
print(f"Testing data shape: {X_test.shape}, {y_test.shape}")


X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

model = Sequential([
    Conv1D(filters=64, kernel_size=1, activation='relu', input_shape=(X_train.shape[1], 1)),
    Dropout(0.2),
    Conv1D(filters=32, kernel_size=1, activation='relu'),
    Flatten(),
    Dense(50, activation='relu'),
    Dense(1)  
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=16, verbose=1)


test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {test_loss}")
print(f"Test Mean Absolute Error: {test_mae}")

y_pred = model.predict(X_test)


plt.figure(figsize=(10, 6))
plt.plot(y_test, label='True Values', color='blue')
plt.title('True Values')
plt.legend()
plt.show()


plt.figure(figsize=(10, 6))
plt.plot(y_pred, label='Predicted Values', color='orange')
plt.title('Predicted Values')
plt.legend()
plt.show()


plt.figure(figsize=(10, 6))
plt.plot(y_test, label='True Values', color='blue')
plt.plot(y_pred, label='Predicted Values', color='orange')
plt.title('Comparison of True and Predicted Values')
plt.legend()
plt.show()
