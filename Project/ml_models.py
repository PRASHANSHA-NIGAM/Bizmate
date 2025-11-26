import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')

def calculate_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'R2': r2
    }

class LinearRegressionModel:
    def __init__(self):
        self.model = LinearRegression()
        self.name = "Linear Regression"
        
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        
    def predict(self, X):
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        return calculate_metrics(y_test, y_pred), y_pred

class RandomForestModel:
    def __init__(self, n_estimators=100, max_depth=10, random_state=42):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1
        )
        self.name = "Random Forest"
        
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        
    def predict(self, X):
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        return calculate_metrics(y_test, y_pred), y_pred
    
    def get_feature_importance(self, feature_names):
        importance = self.model.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        return feature_importance

class LSTMModel:
    def __init__(self, input_shape, units=50):
        self.name = "LSTM"
        self.units = units
        self.model = self._build_model(input_shape)
        
    def _build_model(self, input_shape):
        model = Sequential([
            LSTM(self.units, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(self.units, return_sequences=True),
            Dropout(0.2),
            LSTM(self.units),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model
    
    def train(self, X_train, y_train, epochs=50, batch_size=32, validation_split=0.1):
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stop],
            verbose=0
        )
        
        return history
    
    def predict(self, X):
        predictions = self.model.predict(X, verbose=0)
        return predictions.flatten()
    
    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        return calculate_metrics(y_test, y_pred), y_pred

def train_all_models(X_train_scaled, X_test_scaled, y_train, y_test, feature_cols):
    results = {}
    
    lr_model = LinearRegressionModel()
    lr_model.train(X_train_scaled, y_train)
    lr_metrics, lr_pred = lr_model.evaluate(X_test_scaled, y_test)
    results['Linear Regression'] = {
        'model': lr_model,
        'metrics': lr_metrics,
        'predictions': lr_pred
    }
    
    rf_model = RandomForestModel(n_estimators=100, max_depth=10)
    rf_model.train(X_train_scaled, y_train)
    rf_metrics, rf_pred = rf_model.evaluate(X_test_scaled, y_test)
    results['Random Forest'] = {
        'model': rf_model,
        'metrics': rf_metrics,
        'predictions': rf_pred
    }
    
    return results

def train_lstm_model(X_train, y_train, X_test, y_test, lookback=60):
    from data_preprocessing import prepare_lstm_data
    from sklearn.preprocessing import MinMaxScaler
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    train_data = np.concatenate([y_train, y_test])
    scaled_data = scaler.fit_transform(train_data.values.reshape(-1, 1))
    
    X_lstm, y_lstm = prepare_lstm_data(scaled_data, lookback=lookback)
    
    split_idx = int(len(X_lstm) * 0.8)
    X_train_lstm = X_lstm[:split_idx]
    y_train_lstm = y_lstm[:split_idx]
    X_test_lstm = X_lstm[split_idx:]
    y_test_lstm = y_lstm[split_idx:]
    
    lstm_model = LSTMModel(input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2]), units=50)
    lstm_model.train(X_train_lstm, y_train_lstm, epochs=50, batch_size=32)
    
    y_pred_scaled = lstm_model.predict(X_test_lstm)
    y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_test_actual = scaler.inverse_transform(y_test_lstm.reshape(-1, 1)).flatten()
    
    lstm_metrics = calculate_metrics(y_test_actual, y_pred)
    
    return {
        'model': lstm_model,
        'metrics': lstm_metrics,
        'predictions': y_pred,
        'actual': y_test_actual,
        'scaler': scaler
    }

def predict_future_prices(model, last_data, scaler, days=30, model_type='traditional'):
    predictions = []
    
    if model_type == 'LSTM':
        current_batch = last_data.copy()
        
        for _ in range(days):
            pred = model.predict(current_batch.reshape(1, current_batch.shape[0], current_batch.shape[1]))
            predictions.append(pred[0])
            
            current_batch = np.roll(current_batch, -1, axis=0)
            current_batch[-1] = pred[0]
        
        predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    else:
        current_features = last_data.copy()
        
        for _ in range(days):
            pred = model.predict(current_features.reshape(1, -1))
            predictions.append(pred[0])
            
            current_features = np.roll(current_features, -1)
            current_features[-1] = pred[0]
    
    return np.array(predictions)
