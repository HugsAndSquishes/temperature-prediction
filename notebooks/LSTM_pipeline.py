from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
import os
import pickle

class LSTMPipeline:
    def __init__(self, seq_len=12):
        self.seq_len = seq_len  # Number of months to predict from
        self.scaler = MinMaxScaler()
        self.model = None
        self.X_test = None
        self.y_test = None

    def create_sequences(self, data):
        X, y = [], []
        for i in range(len(data) - self.seq_len):
            X.append(data[i:i+self.seq_len])
            y.append(data[i+self.seq_len])
        return np.array(X), np.array(y)
    
    def set_test_data(self, raw_data):
        data_scaled = self.scaler.transform(raw_data.reshape(-1, 1))
        X, y = self.create_sequences(data_scaled)
        split = int(len(X) * 0.8)
        self.X_test, self.y_test = X[split:], y[split:]

    def fit(self, raw_data):
        data_scaled = self.scaler.fit_transform(raw_data.reshape(-1, 1))
        
        X, y = self.create_sequences(data_scaled)

        split = int(len(X) * 0.8)
        self.X_train, self.y_train = X[:split], y[:split]
        self.X_test, self.y_test = X[split:], y[split:]

        self.model = Sequential()
        self.model.add(LSTM(64, input_shape=(self.seq_len, 1)))
        self.model.add(Dense(1))
        self.model.compile(optimizer='adam', loss='mse')

        # Train
        self.model.fit(self.X_train, self.y_train, epochs=30, batch_size=16, validation_split=0.1)


    def predict(self):
        y_pred = self.model.predict(self.X_test, verbose=0)
        return self.scaler.inverse_transform(y_pred), self.scaler.inverse_transform(self.y_test.reshape(-1, 1))

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
        r2 = r2_score(y_true, y_pred)
        return {'MAE': mae, 'RMSE': rmse, 'R2': r2}
    
    def load(self, model_path, scaler_path=None):
        self.model = load_model(model_path)
        if scaler_path:
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
    
    def save(self, model_path, scaler_path=None):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        self.model.save(model_path, save_format="keras")
        if scaler_path:
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)