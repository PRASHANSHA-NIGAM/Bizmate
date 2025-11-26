import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def fetch_stock_data(ticker, period='2y'):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        
        if df.empty:
            return None, f"No data found for ticker {ticker}"
        
        df = df.reset_index()
        return df, None
    except Exception as e:
        return None, str(e)

def calculate_moving_averages(df, windows=[5, 10, 20, 50]):
    df = df.copy()
    for window in windows:
        df[f'SMA_{window}'] = df['Close'].rolling(window=window).mean()
        df[f'EMA_{window}'] = df['Close'].ewm(span=window, adjust=False).mean()
    return df

def calculate_rsi(df, period=14):
    df = df.copy()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

def calculate_volatility(df, window=20):
    df = df.copy()
    df['Volatility'] = df['Close'].pct_change().rolling(window=window).std() * np.sqrt(252)
    return df

def calculate_macd(df, fast=12, slow=26, signal=9):
    df = df.copy()
    df['EMA_fast'] = df['Close'].ewm(span=fast, adjust=False).mean()
    df['EMA_slow'] = df['Close'].ewm(span=slow, adjust=False).mean()
    df['MACD'] = df['EMA_fast'] - df['EMA_slow']
    df['MACD_Signal'] = df['MACD'].ewm(span=signal, adjust=False).mean()
    df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
    return df

def calculate_bollinger_bands(df, window=20, num_std=2):
    df = df.copy()
    df['BB_Middle'] = df['Close'].rolling(window=window).mean()
    bb_std = df['Close'].rolling(window=window).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * num_std)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * num_std)
    return df

def add_lag_features(df, lags=[1, 2, 3, 5, 7]):
    df = df.copy()
    for lag in lags:
        df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
    return df

def prepare_features(df):
    df = df.copy()
    
    df = calculate_moving_averages(df, windows=[5, 10, 20, 50])
    df = calculate_rsi(df)
    df = calculate_volatility(df)
    df = calculate_macd(df)
    df = calculate_bollinger_bands(df)
    df = add_lag_features(df, lags=[1, 2, 3, 5, 7])
    
    df['Price_Change'] = df['Close'].pct_change()
    df['Price_Range'] = df['High'] - df['Low']
    df['Day_of_Week'] = pd.to_datetime(df['Date']).dt.dayofweek
    df['Month'] = pd.to_datetime(df['Date']).dt.month
    
    df = df.dropna()
    
    return df

def prepare_data_for_training(df, target_col='Close', test_size=0.2):
    df = df.copy()
    
    feature_cols = [col for col in df.columns if col not in ['Date', target_col, 'Dividends', 'Stock Splits']]
    
    X = df[feature_cols]
    y = df[target_col]
    
    split_idx = int(len(df) * (1 - test_size))
    
    X_train = X[:split_idx]
    X_test = X[split_idx:]
    y_train = y[:split_idx]
    y_test = y[split_idx:]
    
    return X_train, X_test, y_train, y_test, feature_cols

def normalize_data(X_train, X_test):
    from sklearn.preprocessing import StandardScaler
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, scaler

def prepare_lstm_data(data, lookback=60, forecast_horizon=1):
    X, y = [], []
    
    for i in range(lookback, len(data) - forecast_horizon + 1):
        X.append(data[i - lookback:i])
        y.append(data[i + forecast_horizon - 1])
    
    return np.array(X), np.array(y)

def get_stock_info(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        return {
            'name': info.get('longName', ticker),
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A'),
            'market_cap': info.get('marketCap', 'N/A'),
            'current_price': info.get('currentPrice', 'N/A'),
        }
    except:
        return {
            'name': ticker,
            'sector': 'N/A',
            'industry': 'N/A',
            'market_cap': 'N/A',
            'current_price': 'N/A',
        }
