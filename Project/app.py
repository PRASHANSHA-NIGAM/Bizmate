import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from data_preprocessing import (
    fetch_stock_data, prepare_features, prepare_data_for_training,
    normalize_data, get_stock_info
)
from ml_models import (
    train_all_models, train_lstm_model, predict_future_prices,
    LinearRegressionModel, RandomForestModel
)

st.set_page_config(
    page_title="Stock Price Prediction",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Stock Price Prediction using Machine Learning")
st.markdown("Predict future stock prices using Linear Regression, Random Forest, and LSTM models")

@st.cache_data
def load_and_prepare_data(ticker, period):
    df, error = fetch_stock_data(ticker, period)
    if error:
        return None, None, error
    
    df_features = prepare_features(df)
    return df, df_features, None

def plot_stock_price(df, ticker):
    fig = go.Figure()
    
    fig.add_trace(go.Candlestick(
        x=df['Date'],
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Price'
    ))
    
    fig.update_layout(
        title=f'{ticker} Stock Price',
        yaxis_title='Price ($)',
        xaxis_title='Date',
        height=500,
        xaxis_rangeslider_visible=False
    )
    
    return fig

def plot_moving_averages(df, ticker):
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['Date'], y=df['Close'],
        mode='lines', name='Close Price',
        line=dict(color='blue', width=2)
    ))
    
    if 'SMA_20' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['Date'], y=df['SMA_20'],
            mode='lines', name='SMA 20',
            line=dict(color='orange', dash='dash')
        ))
    
    if 'SMA_50' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['Date'], y=df['SMA_50'],
            mode='lines', name='SMA 50',
            line=dict(color='green', dash='dash')
        ))
    
    if 'EMA_20' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['Date'], y=df['EMA_20'],
            mode='lines', name='EMA 20',
            line=dict(color='red', dash='dot')
        ))
    
    fig.update_layout(
        title=f'{ticker} Price with Moving Averages',
        yaxis_title='Price ($)',
        xaxis_title='Date',
        height=500
    )
    
    return fig

def plot_technical_indicators(df):
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('RSI', 'MACD', 'Volatility'),
        vertical_spacing=0.1,
        row_heights=[0.33, 0.33, 0.33]
    )
    
    if 'RSI' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['Date'], y=df['RSI'],
            mode='lines', name='RSI',
            line=dict(color='purple')
        ), row=1, col=1)
        
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=1, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=1, col=1)
    
    if 'MACD' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['Date'], y=df['MACD'],
            mode='lines', name='MACD',
            line=dict(color='blue')
        ), row=2, col=1)
        
        fig.add_trace(go.Scatter(
            x=df['Date'], y=df['MACD_Signal'],
            mode='lines', name='Signal',
            line=dict(color='orange')
        ), row=2, col=1)
    
    if 'Volatility' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['Date'], y=df['Volatility'],
            mode='lines', name='Volatility',
            line=dict(color='green')
        ), row=3, col=1)
    
    fig.update_layout(height=800, showlegend=True)
    fig.update_xaxes(title_text="Date", row=3, col=1)
    
    return fig

def plot_predictions(y_test, predictions, dates, model_name):
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=dates, y=y_test,
        mode='lines', name='Actual',
        line=dict(color='blue', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=dates, y=predictions,
        mode='lines', name='Predicted',
        line=dict(color='red', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title=f'{model_name} - Actual vs Predicted Prices',
        yaxis_title='Price ($)',
        xaxis_title='Date',
        height=400
    )
    
    return fig

def plot_future_predictions(last_date, future_predictions, days):
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=days)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=future_dates, y=future_predictions,
        mode='lines+markers', name='Predicted Prices',
        line=dict(color='green', width=2),
        marker=dict(size=6)
    ))
    
    fig.update_layout(
        title=f'Future Price Predictions ({days} Days)',
        yaxis_title='Predicted Price ($)',
        xaxis_title='Date',
        height=400
    )
    
    return fig

def plot_model_comparison(results):
    models = list(results.keys())
    rmse_values = [results[model]['metrics']['RMSE'] for model in models]
    mape_values = [results[model]['metrics']['MAPE'] for model in models]
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('RMSE Comparison', 'MAPE Comparison')
    )
    
    fig.add_trace(go.Bar(
        x=models, y=rmse_values,
        name='RMSE',
        marker_color='lightblue'
    ), row=1, col=1)
    
    fig.add_trace(go.Bar(
        x=models, y=mape_values,
        name='MAPE (%)',
        marker_color='lightcoral'
    ), row=1, col=2)
    
    fig.update_layout(height=400, showlegend=False)
    fig.update_yaxes(title_text="RMSE", row=1, col=1)
    fig.update_yaxes(title_text="MAPE (%)", row=1, col=2)
    
    return fig

with st.sidebar:
    st.header("Configuration")
    
    ticker = st.text_input("Stock Ticker", value="AAPL", help="Enter stock symbol (e.g., AAPL, GOOGL, MSFT)")
    
    period_options = {
        '1 Month': '1mo',
        '3 Months': '3mo',
        '6 Months': '6mo',
        '1 Year': '1y',
        '2 Years': '2y',
        '5 Years': '5y'
    }
    period_label = st.selectbox("Time Period", list(period_options.keys()), index=4)
    period = period_options[period_label]
    
    st.divider()
    
    st.subheader("Model Settings")
    test_size = st.slider("Test Set Size (%)", 10, 40, 20) / 100
    
    train_lstm = st.checkbox("Include LSTM Model", value=True, help="LSTM training takes longer but often provides better predictions")
    
    if train_lstm:
        lstm_lookback = st.slider("LSTM Lookback Period", 30, 90, 60, help="Number of days to look back for LSTM predictions")
    
    train_models_btn = st.button("🚀 Train Models", type="primary")

if ticker:
    ticker = ticker.upper()
    
    with st.spinner(f"Fetching data for {ticker}..."):
        df, df_features, error = load_and_prepare_data(ticker, period)
    
    if error:
        st.error(f"Error: {error}")
        st.info("Please check the ticker symbol and try again.")
    else:
        stock_info = get_stock_info(ticker)
        
        st.success(f"✅ Data loaded successfully for {stock_info['name']}")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Sector", stock_info['sector'])
        with col2:
            st.metric("Industry", stock_info['industry'])
        with col3:
            if isinstance(stock_info['market_cap'], (int, float)):
                st.metric("Market Cap", f"${stock_info['market_cap']:,.0f}")
            else:
                st.metric("Market Cap", stock_info['market_cap'])
        with col4:
            if isinstance(stock_info['current_price'], (int, float)):
                st.metric("Current Price", f"${stock_info['current_price']:.2f}")
            else:
                st.metric("Current Price", stock_info['current_price'])
        
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Overview", "📈 Technical Analysis", "🤖 Model Training", "🔮 Future Predictions"])
        
        with tab1:
            st.subheader("Historical Stock Data")
            
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(plot_stock_price(df, ticker), width='stretch')
            with col2:
                st.plotly_chart(plot_moving_averages(df_features, ticker), width='stretch')
            
            with st.expander("📋 View Raw Data"):
                st.dataframe(df.tail(100), width='stretch')
            
            st.subheader("Basic Statistics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Average Price", f"${df['Close'].mean():.2f}")
                st.metric("Min Price", f"${df['Close'].min():.2f}")
            with col2:
                st.metric("Max Price", f"${df['Close'].max():.2f}")
                st.metric("Std Deviation", f"${df['Close'].std():.2f}")
            with col3:
                st.metric("Total Trading Days", len(df))
                st.metric("Avg Volume", f"{df['Volume'].mean():,.0f}")
        
        with tab2:
            st.subheader("Technical Indicators")
            st.plotly_chart(plot_technical_indicators(df_features), width='stretch')
            
            with st.expander("📋 View Technical Indicators Data"):
                tech_cols = ['Date', 'Close', 'SMA_20', 'SMA_50', 'EMA_20', 'RSI', 'MACD', 'Volatility']
                available_cols = [col for col in tech_cols if col in df_features.columns]
                st.dataframe(df_features[available_cols].tail(50), width='stretch')
        
        with tab3:
            if train_models_btn:
                with st.spinner("Training models... This may take a few minutes."):
                    try:
                        X_train, X_test, y_train, y_test, feature_cols = prepare_data_for_training(
                            df_features, target_col='Close', test_size=test_size
                        )
                        
                        X_train_scaled, X_test_scaled, scaler = normalize_data(X_train, X_test)
                        
                        progress_text = st.empty()
                        
                        progress_text.text("Training Linear Regression and Random Forest models...")
                        results = train_all_models(X_train_scaled, X_test_scaled, y_train, y_test, feature_cols)
                        
                        if train_lstm:
                            progress_text.text("Training LSTM model... This may take a few minutes.")
                            lookback = lstm_lookback if 'lstm_lookback' in locals() else 60
                            lstm_result = train_lstm_model(X_train, y_train, X_test, y_test, lookback=lookback)
                            results['LSTM'] = lstm_result
                            st.session_state['lstm_lookback'] = lookback
                        
                        progress_text.empty()
                        
                        st.session_state['results'] = results
                        st.session_state['y_test'] = y_test
                        st.session_state['y_train'] = y_train
                        st.session_state['X_test_scaled'] = X_test_scaled
                        st.session_state['scaler'] = scaler
                        st.session_state['df_features'] = df_features
                        st.session_state['feature_cols'] = feature_cols
                        st.session_state['train_lstm'] = train_lstm
                        
                        st.success("✅ Models trained successfully!")
                        
                    except Exception as e:
                        st.error(f"Error training models: {str(e)}")
            
            if 'results' in st.session_state:
                st.subheader("Model Performance Comparison")
                
                results = st.session_state['results']
                y_test = st.session_state['y_test']
                
                st.plotly_chart(plot_model_comparison(results), width='stretch')
                
                metrics_df = pd.DataFrame({
                    'Model': list(results.keys()),
                    'RMSE': [results[model]['metrics']['RMSE'] for model in results.keys()],
                    'MAE': [results[model]['metrics']['MAE'] for model in results.keys()],
                    'MAPE (%)': [results[model]['metrics']['MAPE'] for model in results.keys()],
                    'R² Score': [results[model]['metrics']['R2'] for model in results.keys()]
                })
                
                st.dataframe(metrics_df.style.highlight_min(subset=['RMSE', 'MAE', 'MAPE (%)'], color='lightgreen')
                           .highlight_max(subset=['R² Score'], color='lightgreen'), width='stretch')
                
                st.subheader("Prediction Visualizations")
                
                test_dates = df_features.iloc[-len(y_test):]['Date'].values
                
                for model_name in results.keys():
                    with st.expander(f"📊 {model_name} Predictions"):
                        predictions = results[model_name]['predictions']
                        
                        if model_name == 'LSTM' and 'actual' in results[model_name]:
                            actual_values = results[model_name]['actual']
                            lstm_dates = df_features.iloc[-len(actual_values):]['Date'].values
                            st.plotly_chart(plot_predictions(actual_values, predictions, lstm_dates, model_name), width='stretch')
                        else:
                            st.plotly_chart(plot_predictions(y_test.values, predictions, test_dates, model_name), width='stretch')
                
                if 'Random Forest' in results:
                    with st.expander("🌲 Random Forest Feature Importance"):
                        rf_model = results['Random Forest']['model']
                        feature_importance = rf_model.get_feature_importance(st.session_state['feature_cols'])
                        
                        fig = px.bar(
                            feature_importance.head(15),
                            x='importance',
                            y='feature',
                            orientation='h',
                            title='Top 15 Most Important Features'
                        )
                        st.plotly_chart(fig, width='stretch')
            else:
                st.info("👈 Click 'Train Models' in the sidebar to start training")
        
        with tab4:
            if 'results' in st.session_state:
                st.subheader("Future Price Predictions")
                
                col1, col2 = st.columns([1, 3])
                
                with col1:
                    prediction_days = st.slider("Prediction Horizon (Days)", 5, 90, 30)
                    selected_model = st.selectbox("Select Model", list(st.session_state['results'].keys()))
                
                with col2:
                    if st.button("🔮 Predict Future Prices"):
                        with st.spinner("Generating predictions..."):
                            try:
                                model = st.session_state['results'][selected_model]['model']
                                future_preds = None
                                
                                if selected_model == 'LSTM':
                                    st.warning("⚠️ LSTM future predictions are experimental and may not be as reliable as historical predictions.")
                                    lstm_scaler = st.session_state['results']['LSTM']['scaler']
                                    y_train = st.session_state['y_train']
                                    y_test = st.session_state['y_test']
                                    lstm_lookback = st.session_state.get('lstm_lookback', 60)
                                    
                                    full_data = np.concatenate([y_train.values, y_test.values])
                                    scaled_data = lstm_scaler.transform(full_data.reshape(-1, 1))
                                    
                                    if len(scaled_data) < lstm_lookback:
                                        st.error(f"Not enough data for LSTM prediction. Need at least {lstm_lookback} data points.")
                                    else:
                                        last_sequence = scaled_data[-lstm_lookback:]
                                        
                                        future_preds = predict_future_prices(
                                            model, last_sequence, lstm_scaler, days=prediction_days, model_type='LSTM'
                                        )
                                else:
                                    X_test_scaled = st.session_state['X_test_scaled']
                                    last_features = X_test_scaled[-1]
                                    
                                    future_preds = predict_future_prices(
                                        model, last_features, None, days=prediction_days, model_type='traditional'
                                    )
                                
                                if future_preds is not None:
                                    last_date = df_features['Date'].iloc[-1]
                                    
                                    st.plotly_chart(plot_future_predictions(last_date, future_preds, prediction_days), width='stretch')
                                    
                                    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=prediction_days)
                                    future_df = pd.DataFrame({
                                        'Date': future_dates,
                                        'Predicted Price': future_preds
                                    })
                                    
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("Starting Price", f"${future_preds[0]:.2f}")
                                    with col2:
                                        st.metric("Ending Price", f"${future_preds[-1]:.2f}")
                                    with col3:
                                        change_pct = ((future_preds[-1] - future_preds[0]) / future_preds[0]) * 100
                                        st.metric("Expected Change", f"{change_pct:.2f}%")
                                    
                                    with st.expander("📋 View Future Predictions Table"):
                                        st.dataframe(future_df, width='stretch')
                                
                            except Exception as e:
                                st.error(f"Error generating predictions: {str(e)}")
            else:
                st.info("⚠️ Please train models first in the 'Model Training' tab")

st.divider()
st.markdown("""
### About this Application
This stock price prediction application uses machine learning to forecast future stock prices based on historical data and technical indicators.

**Models Used:**
- **Linear Regression**: Simple baseline model for trend prediction
- **Random Forest**: Ensemble model for capturing complex patterns
- **LSTM (Long Short-Term Memory)**: Deep learning model for time-series prediction

**Features:**
- Moving Averages (SMA, EMA)
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- Volatility measures
- Price lag features

**Note**: Stock predictions are for educational purposes only and should not be used as financial advice.
""")
