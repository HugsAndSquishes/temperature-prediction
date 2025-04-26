from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
import os
import pickle
import pandas as pd


class LSTMPipeline:
    def __init__(self, seq_len=12):
        self.seq_len = seq_len
        self.temp_scaler = MinMaxScaler()  # Only for temperature
        self.model = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

    def create_sequences(self, data, labels):
        X, y = [], []
        for i in range(len(data) - self.seq_len):
            X.append(data[i:i+self.seq_len])
            y.append(labels[i+self.seq_len])
        return np.array(X), np.array(y)

    def fit(self, temp_data, country_labels=None):
        temp_scaled = self.temp_scaler.fit_transform(temp_data.reshape(-1, 1))
        
        if country_labels is not None:
            if isinstance(country_labels, pd.DataFrame) or isinstance(country_labels, pd.Series):
                country_labels = country_labels.values.reshape(-1, 1) 

            self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            country_encoded = self.encoder.fit_transform(country_labels)

            X_combined = np.concatenate([temp_scaled, country_encoded], axis=1)
        else:
            X_combined = temp_scaled 

        

        X_combined = X_combined.astype(np.float32)
        temp_scaled = temp_scaled.astype(np.float32) 

        X, y = self.create_sequences(X_combined, temp_scaled)

        
        split = int(len(X) * 0.8)
        self.X_train, self.y_train = X[:split], y[:split]
        self.X_test, self.y_test = X[split:], y[split:]

        self.model = Sequential()
        self.model.add(LSTM(64, input_shape=(self.seq_len, X.shape[2])))
        self.model.add(Dense(1))
        self.model.compile(optimizer='adam', loss='mse')

        print(f"Input shape to LSTM: {self.X_train.shape}")
        print(f"Output shape: {self.y_train.shape}")

        self.model.fit(self.X_train, self.y_train, epochs=20, batch_size=16, validation_split=0.1)

    def predict(self):
        y_pred = self.model.predict(self.X_test, verbose=0)
        return self.temp_scaler.inverse_transform(y_pred), self.temp_scaler.inverse_transform(self.y_test.reshape(-1, 1))

    def plot_results(self):
        y_pred, y_true = self.predict()
        plt.figure(figsize=(10, 5))
        plt.plot(y_true, label='True')
        plt.plot(y_pred, label='Predicted')
        plt.legend()
        plt.title("LSTM Prediction vs True")
        plt.show()

    def evaluate(self):
        y_pred, y_true = self.predict()
        mae = mean_absolute_error(y_true, y_pred)
        rmse = root_mean_squared_error(y_true, y_pred)
        mse = mean_squared_error(y_true, linear_regression_y_pred)
        r2 = r2_score(y_true, y_pred)
        return {'MAE': mae, 'RMSE': rmse, 'R2': r2m, 'MSE': mean_squared_error}

    def load(self, model_path, scaler_path=None):
        self.model = load_model(model_path)
        if scaler_path:
            with open(scaler_path, 'rb') as f:
                self.temp_scaler = pickle.load(f)
    
    def save(self, model_path, scaler_path=None):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        self.model.save(model_path, save_format="keras")
        if scaler_path:
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.temp_scaler, f)