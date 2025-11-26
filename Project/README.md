# Stock Price Prediction Application

## Overview

This is a stock price prediction application built with Streamlit that leverages multiple machine learning models to forecast future stock prices. The application fetches real-time stock data from Yahoo Finance and applies three different predictive models: Linear Regression, Random Forest, and LSTM (Long Short-Term Memory) neural networks. Users can visualize historical stock data with candlestick charts, analyze technical indicators, and compare model predictions to make informed investment decisions.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture

**Technology Choice: Streamlit**
- **Rationale**: Streamlit provides a rapid development framework for data science applications with minimal frontend code
- **Key Features**: 
  - Interactive widgets for user input (ticker selection, date ranges)
  - Real-time data visualization using Plotly charts
  - Caching mechanism (`@st.cache_data`) to optimize data loading performance
- **Design Pattern**: Single-page application with wide layout configuration for better chart visualization
- **Pros**: Fast prototyping, Python-native, built-in state management
- **Cons**: Limited customization compared to React/Vue frameworks

### Data Processing Pipeline

**Modular Architecture Pattern**
- **Component**: `data_preprocessing.py` handles all data acquisition and feature engineering
- **Key Responsibilities**:
  - Stock data fetching via yfinance API
  - Technical indicator calculations (RSI, MACD, Bollinger Bands, Moving Averages)
  - Volatility metrics computation
  - Feature normalization for model input
- **Design Decision**: Separation of concerns - data operations isolated from ML logic and UI
- **Alternative Considered**: All-in-one script approach rejected due to maintainability concerns

### Machine Learning Architecture

**Multi-Model Ensemble Approach**
- **Component**: `ml_models.py` implements three distinct prediction models
- **Models Implemented**:
  1. **Linear Regression**: Baseline model for linear trend capture
  2. **Random Forest**: Ensemble method for non-linear pattern recognition (100 estimators, max depth 10)
  3. **LSTM Neural Network**: Deep learning model for sequential time-series patterns using TensorFlow/Keras
- **Rationale**: Multiple models provide diverse perspectives and allow comparison of prediction accuracy
- **Evaluation Metrics**: RMSE, MAE, MAPE, and R² score for comprehensive performance assessment
- **Training Strategy**: Separate training pipelines for each model with early stopping for LSTM to prevent overfitting

### Data Flow Architecture

**Pipeline Design**:
1. User inputs ticker symbol and time period
2. Cache layer checks for existing data to avoid redundant API calls
3. Data preprocessing module fetches and enriches raw stock data
4. Feature engineering creates technical indicators
5. Data normalization prepares features for model consumption
6. All models train on prepared dataset
7. Predictions generated and visualized alongside historical data

### Visualization Layer

**Technology: Plotly**
- **Rationale**: Interactive charts with zoom, pan, and hover capabilities enhance user experience
- **Chart Types**:
  - Candlestick charts for OHLC (Open, High, Low, Close) data
  - Line charts for predictions and moving averages
  - Subplots for comparative model analysis
- **Pros**: Rich interactivity, professional financial charting, web-native
- **Cons**: Larger bundle size compared to static charts

## External Dependencies

### Data Sources

**Yahoo Finance API (via yfinance library)**
- **Purpose**: Real-time and historical stock price data retrieval
- **Data Retrieved**: OHLC prices, volume, dividends, stock splits
- **Period Support**: Flexible time ranges (default: 2 years)
- **Rate Limits**: Subject to Yahoo Finance API restrictions
- **Error Handling**: Graceful fallback with error messages when ticker is invalid or data unavailable

### Machine Learning Frameworks

**scikit-learn**
- **Purpose**: Traditional ML models (Linear Regression, Random Forest)
- **Components Used**: 
  - Model classes: `LinearRegression`, `RandomForestRegressor`
  - Metrics: `mean_squared_error`, `mean_absolute_error`, `r2_score`
- **Version Considerations**: Compatible with standard sklearn API

**TensorFlow/Keras**
- **Purpose**: Deep learning LSTM model implementation
- **Architecture**: Sequential model with LSTM layers and Dropout for regularization
- **Training Features**: Early stopping callbacks to optimize training time and prevent overfitting
- **GPU Support**: Compatible with CUDA for accelerated training (optional)

### Data Processing Libraries

**pandas**
- **Purpose**: Core data manipulation and time-series operations
- **Key Operations**: DataFrame transformations, rolling window calculations, date handling

**numpy**
- **Purpose**: Numerical computations and array operations
- **Use Cases**: Mathematical transformations, statistical calculations, data normalization

### Visualization Libraries

**Plotly (plotly.graph_objects and plotly.express)**
- **Purpose**: Interactive financial charts and visualizations
- **Integration**: Embedded directly in Streamlit application
- **Chart Types**: Candlestick, line plots, subplots for multi-model comparison

### Supporting Libraries

**warnings** (Python standard library)
- **Purpose**: Suppress deprecation warnings from ML libraries to maintain clean UI

**datetime** (Python standard library)
- **Purpose**: Date range calculations and time-series indexing
