import pandas as pd

file_path = "https://raw.githubusercontent.com/Zero-Asif/data/refs/heads/main/AirPassengers.csv"  
data = pd.read_csv(file_path)

print(data.columns)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import keras_tuner as kt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def load_data(file_path):
    
    data = pd.read_csv(file_path)

    
    print("Column names in the dataset:", data.columns)

    
    if '#Passengers' not in data.columns:
        raise ValueError("'#Passengers' column is missing in the dataset.")
    
    data['Month'] = pd.to_datetime(data['Month'])  
    data.set_index('Month', inplace=True)         

    
    y = data['#Passengers'].values
    X = data.index.astype('int64').to_numpy().reshape(-1, 1)  
    
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    y = scaler.fit_transform(y.reshape(-1, 1)).flatten()

    
    X = X.reshape((X.shape[0], 1, X.shape[1]))
    
    return train_test_split(X, y, test_size=0.2, random_state=42)

def build_model(hp):
    model = keras.Sequential()

    
    for i in range(hp.Int("num_lstm_layers", 1, 3)):
        model.add(
            layers.LSTM(
                units=hp.Int(f"units_{i}", min_value=32, max_value=256, step=32),
                activation=hp.Choice("activation", ["relu", "elu", "sigmoid"]),
                return_sequences=True if i < hp.Int("num_lstm_layers", 1, 3) - 1 else False
            )
        )
        if hp.Boolean(f"add_dropout_{i}"):
            model.add(layers.Dropout(rate=hp.Float(f"dropout_rate_{i}", 0.1, 0.5, step=0.1)))

    
    for i in range(hp.Int("num_dense_layers", 1, 3)):
        model.add(
            layers.Dense(
                units=hp.Int(f"dense_units_{i}", min_value=32, max_value=256, step=32),
                activation=hp.Choice(f"dense_activation_{i}", ["relu", "elu", "sigmoid"])
            )
        )
        if hp.Boolean(f"add_dense_dropout_{i}"):
            model.add(layers.Dropout(rate=hp.Float(f"dense_dropout_rate_{i}", 0.1, 0.5, step=0.1)))

    
    model.add(layers.Dense(1, activation="linear"))

    
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=hp.Choice("learning_rate", [1e-3, 1e-2, 1e-4])
        ),
        loss=hp.Choice("loss", ["mse", "mae"]),
        metrics=["mae"]
    )

    return model


file_path = "https://raw.githubusercontent.com/Zero-Asif/data/refs/heads/main/AirPassengers.csv"  
X_train, X_test, y_train, y_test = load_data(file_path)


tuner = kt.Hyperband(
    build_model,
    objective="val_mae",
    max_epochs=10,
    factor=3,
    directory="my_dir",
    project_name="lstm_tuning"
)


stop_early = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)


tuner.search(X_train, y_train, epochs=50, validation_split=0.2, callbacks=[stop_early])

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print("Best hyperparameters:")
for key, value in best_hps.values.items():
    print(f"{key}: {value}")


model = tuner.hypermodel.build(best_hps)
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, callbacks=[stop_early])


eval_result = model.evaluate(X_test, y_test)
print(f"Test Loss: {eval_result[0]}, Test MAE: {eval_result[1]}")
