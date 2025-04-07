import numpy as np
import pandas as pd
import streamlit as st
import datetime as dt
import yfinance as yf
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import os

# Load the trained model
model = load_model('stockpricemodel.keras')

st.title("Stock Price Prediction App")

# User input for stock symbol
stock = st.text_input("Enter Stock Symbol (e.g., NVDA)", "NVDA")

# Define start and end dates
start = dt.datetime(2000, 1, 1)
end = dt.datetime.now()

# ✅ Error Handling for Invalid Symbols
try:
    df = yf.download(stock, start=start, end=end)
    if df.empty:
        st.error(f"❌ No data found for symbol '{stock}'. Please check the symbol and try again.")
        st.stop()
except Exception as e:
    st.error(f"❌ Error retrieving data for symbol '{stock}': {e}")
    st.stop()

# Display dataset summary
st.subheader("Stock Data Summary")
st.write(df.describe())

# Calculate Exponential Moving Averages (EMA)
df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
df['EMA_100'] = df['Close'].ewm(span=100, adjust=False).mean()
df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()

# Candlestick Chart with 20 & 50 EMA
st.subheader("Closing Price with 20 & 50 Days EMA")
fig1 = go.Figure()
fig1.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Candlestick'))
fig1.add_trace(go.Scatter(x=df.index, y=df['EMA_20'], mode='lines', name='EMA 20', line=dict(color='green')))
fig1.add_trace(go.Scatter(x=df.index, y=df['EMA_50'], mode='lines', name='EMA 50', line=dict(color='red')))
fig1.update_layout(xaxis_rangeslider_visible=False)
st.plotly_chart(fig1)

# Candlestick Chart with 100 & 200 EMA
st.subheader("Closing Price with 100 & 200 Days EMA")
fig2 = go.Figure()
fig2.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Candlestick'))
fig2.add_trace(go.Scatter(x=df.index, y=df['EMA_100'], mode='lines', name='EMA 100', line=dict(color='blue')))
fig2.add_trace(go.Scatter(x=df.index, y=df['EMA_200'], mode='lines', name='EMA 200', line=dict(color='purple')))
fig2.update_layout(xaxis_rangeslider_visible=False)
st.plotly_chart(fig2)

# Data Preparation
data_training = df[['Close']][:int(len(df) * 0.70)]
data_testing = df[['Close']][int(len(df) * 0.70):]
scaler = MinMaxScaler(feature_range=(0, 1))
data_training_array = scaler.fit_transform(data_training)

# Prepare test data
past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.transform(final_df)

x_test, y_test = [], []
for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i - 100:i])
    y_test.append(input_data[i, 0])
x_test, y_test = np.array(x_test), np.array(y_test)

# Make predictions
y_predicted = model.predict(x_test)

# Reverse scaling
scale_factor = 1 / scaler.scale_[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

# Extract corresponding dates for the test dataset
test_dates = df.index[-len(y_test):]

# Plot Predictions
st.subheader("Prediction vs Original Trend")
fig3 = go.Figure()
fig3.add_trace(go.Scatter(
    x=test_dates, y=y_test, mode='lines', name="Actual Price", line=dict(color="blue")
))
fig3.add_trace(go.Scatter(
    x=test_dates, y=y_predicted.flatten(), mode='lines', name="Predicted Price", line=dict(color="orange")
))

fig3.update_layout(title="Stock Price Prediction",
                   xaxis_title="Date",
                   yaxis_title="Price",
                   xaxis_rangeslider_visible=False)
st.plotly_chart(fig3)

# Get last closing price correctly as float
current_price = df['Close'].iloc[-1]
current_price = float(current_price) if isinstance(current_price, (pd.Series, pd.DataFrame)) else current_price

# Get current date (the last date in our dataset)
current_date = df.index[-1]

# Get the prediction for the current day (the last prediction in y_predicted)
current_day_prediction = float(y_predicted[-1][0])

# Calculate the error rate and accuracy for the current day prediction
current_day_error_rate = abs((current_day_prediction - current_price) / current_price) * 100
current_day_accuracy = 100 - current_day_error_rate

# 🚀 Predict the next 10 days recursively
last_100_days = input_data[-100:]
future_predictions = []
current_input = last_100_days.reshape(1, 100, 1)

# Predict next 10 days
previous_price = y_test[-1]
for _ in range(10):
    next_prediction = model.predict(current_input)[0][0] * scale_factor
    
    # ✅ Add realistic fluctuation (including falls)
    noise = np.random.uniform(-5, 5)  # Simulate realistic fluctuation
    next_prediction += noise
    
    # ✅ Limit the difference between consecutive prices to ±20
    if len(future_predictions) > 0:
        next_prediction = np.clip(next_prediction, future_predictions[-1] - 5.043, future_predictions[-1] + 5.597)
    else:
        next_prediction = np.clip(next_prediction, previous_price - 5.673, previous_price + 5.068)
    
    future_predictions.append(next_prediction)
    
    # Update input for next prediction
    next_scaled = next_prediction / scale_factor
    current_input = np.append(current_input[:, 1:, :], [[[next_scaled]]], axis=1)

# Reverse scaling for target values
future_predictions = np.array(future_predictions)

# Get the current time
now = dt.datetime.now()
current_time = now.time()
midnight = dt.time(0, 1)  # 12:01 AM

# Generate dates for predictions
# If current time is after midnight (00:01), shift the predictions forward
if current_time >= midnight:
    # Start future dates from today (current day)
    future_dates = pd.date_range(start=now.date(), periods=6)
    # Use only future predictions for the table
    target_dates = list(future_dates)
    target_prices = list(future_predictions[:6])
    # The first prediction in the table is the current day prediction
    top_prediction = target_prices[0]
else:
    # Include current day + future days
    future_dates = pd.date_range(start=df.index[-1] + dt.timedelta(days=1), periods=10)
    target_dates = [current_date] + list(future_dates[:5])
    target_prices = [current_day_prediction] + list(future_predictions[:5])
    # The first prediction in the table is the current day prediction
    top_prediction = current_day_prediction

# Create the table
target_df = pd.DataFrame({
    'Date': [d.strftime('%Y-%m-%d') for d in target_dates],
    'Predicted Target Price': target_prices
})

# Display table
st.subheader("Predicted Prices (Next 6 Days)")
st.write(target_df)

# Get ticker information for currency
ticker_info = yf.Ticker(stock).info
currency = ticker_info.get("currency", "USD") 
currency_symbols = {
    "USD": "$",
    "INR": "₹",
    "EUR": "€",
    "GBP": "£",
    "JPY": "¥",
    "AUD": "A$",
    "CAD": "C$",
    "CHF": "CHF",
    "SGD": "S$",
    "HKD": "HK$",
}

# Get the appropriate currency symbol
currency_symbol = currency_symbols.get(currency, currency)
st.subheader("📈 Single-Day Gain/Loss Calculator")

formatted_price = f"Current Stock Price: *{currency_symbol}{current_price}*"
st.write(formatted_price)

shares_bought = st.number_input("Enter the number of shares you want to buy:", min_value=1, step=1, value=1)

# Determine which prediction to use based on time
if current_time >= midnight:
    # After midnight, use first future prediction
    prediction_for_calc = future_predictions[0]
    day_label = "today"
else:
    # Before midnight, use current day prediction
    prediction_for_calc = current_day_prediction
    day_label = "current day"

# Calculate profit/loss
profit_loss = (prediction_for_calc - current_price) * shares_bought

if profit_loss > 0:
    st.success(f"🎉 Expected *Profit* for {day_label}: *{currency_symbol}{profit_loss:.2f}*")
elif profit_loss < 0:
    st.error(f"⚠ Expected *Loss* for {day_label}: *{currency_symbol}{profit_loss:.2f}*")
else:
    st.info(f"No gain or loss expected for {day_label}.")

# Display model accuracy metrics for current day prediction
st.subheader("🎯 Model Accuracy Metrics")
st.write(f"Error Rate for Current Day Prediction: *{current_day_error_rate:.2f}%*")
st.write(f"Accuracy for Current Day Prediction: *{current_day_accuracy:.2f}%*")

# Add visualization for accuracy
if current_day_accuracy >= 90:
    st.success("🌟 High Accuracy Prediction")
elif current_day_accuracy >= 75:
    st.info("✅ Good Accuracy Prediction")
else:
    st.warning("⚠ Lower Accuracy Prediction - Use caution")

# ✅ Download dataset
csv_file_path = f"{stock}_dataset.csv"
df.to_csv(csv_file_path)
st.download_button(label="Download Dataset as CSV", data=open(csv_file_path, 'rb'), file_name=csv_file_path, mime='text/csv')

# Get previous 30 days of historical data and predictions
last_30_days_data = df.tail(30).copy()

# Only calculate predictions for days with 100 previous days available
if len(df) >= 130:  # Need at least 100 + 30 days
    # Generate predictions for the last 30 days
    historical_dates = last_30_days_data.index
    
    # For each of the 30 days, get actual price and predicted price
    historical_actual_prices = last_30_days_data['Close'].values
    
    # Extract the corresponding 30 days of predictions from y_predicted if available
    if len(y_predicted) >= 30:
        # Make sure the predictions are flattened to 1D array
        historical_pred_prices = y_predicted[-30:].flatten()
    else:
        # If we don't have enough predictions, use what we have and pad with NaN
        padding_length = 30 - len(y_predicted)
        historical_pred_prices = np.concatenate([
            np.array([np.nan] * padding_length),
            y_predicted.flatten()
        ])
    
    # Calculate error rates and accuracies for historical predictions
    historical_error_rates = []
    historical_accuracies = []
    
    for actual, pred in zip(historical_actual_prices, historical_pred_prices):
        if not np.isnan(pred):
            error_rate = abs((pred - actual) / actual) * 100
            accuracy = 100 - error_rate
        else:
            error_rate = np.nan
            accuracy = np.nan
        
        historical_error_rates.append(error_rate)
        historical_accuracies.append(accuracy)
    
    # Convert all data to 1D arrays or lists to avoid the ValueError
    historical_df = pd.DataFrame({
        'Date': [date for date in historical_dates],
        'Actual_Price': historical_actual_prices.tolist(),
        'Predicted_Price': historical_pred_prices.tolist(),
        'Error_Rate_Percent': historical_error_rates,
        'Accuracy_Percent': historical_accuracies
    })
    
    # Display historical prediction performance
    st.subheader("Historical Prediction Performance (Last 30 Days)")
    
    # Create a chart showing actual vs predicted prices
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Scatter(
        x=historical_df['Date'], 
        y=historical_df['Actual_Price'],
        mode='lines',
        name='Actual Price',
        line=dict(color='blue')
    ))
    fig_hist.add_trace(go.Scatter(
        x=historical_df['Date'], 
        y=historical_df['Predicted_Price'],
        mode='lines',
        name='Predicted Price',
        line=dict(color='orange')
    ))
    
    fig_hist.update_layout(
        title="Historical Actual vs Predicted Prices (Last 30 Days)",
        xaxis_title="Date",
        yaxis_title=f"Price ({currency})"
    )
    
    st.plotly_chart(fig_hist)
    
    # Show average historical accuracy
    valid_accuracies = [acc for acc in historical_accuracies if not np.isnan(acc)]
    if valid_accuracies:
        avg_historical_accuracy = np.mean(valid_accuracies)
        st.write(f"Average prediction accuracy over last 30 days: *{avg_historical_accuracy:.2f}%*")
else:
    st.warning("Not enough historical data to calculate 30-day prediction history.")
    # Create empty historical dataframe for the record
    historical_df = pd.DataFrame({
        'Date': [],
        'Actual_Price': [],
        'Predicted_Price': [],
        'Error_Rate_Percent': [],
        'Accuracy_Percent': []
    })

# ✅ Save prediction data to CSV
# Create a unique filename based on the stock symbol
prediction_data_filename = f"{stock}_prediction_log.csv"

# Create prediction record
prediction_timestamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
prediction_record = {
    'Timestamp': prediction_timestamp,
    'Stock': stock,
    'Current_Price': current_price,
    'Current_Day_Prediction': top_prediction,  # Using the top prediction from the table
    'Error_Rate_Percent': current_day_error_rate,  # Add error rate
    'Accuracy_Percent': current_day_accuracy  # Add accuracy percentage
}

# For future dates
for i, (date, price) in enumerate(zip(target_dates, target_prices)):
    day_number = i+1
    prediction_record[f'Date_{day_number}'] = date.strftime('%Y-%m-%d') if isinstance(date, dt.datetime) else date
    prediction_record[f'Price_{day_number}'] = price

# Add historical prediction data (last 30 days)
if not historical_df.empty:
    for i in range(min(30, len(historical_df))):
        if i < len(historical_df):
            date_str = historical_df['Date'].iloc[i].strftime('%Y-%m-%d') if isinstance(historical_df['Date'].iloc[i], dt.datetime) else str(historical_df['Date'].iloc[i])
            prediction_record[f'Hist_Date_{i+1}'] = date_str
            prediction_record[f'Hist_Actual_{i+1}'] = historical_df['Actual_Price'].iloc[i]
            prediction_record[f'Hist_Predicted_{i+1}'] = historical_df['Predicted_Price'].iloc[i]
            prediction_record[f'Hist_Accuracy_{i+1}'] = historical_df['Accuracy_Percent'].iloc[i]

# Check if file exists to append or create new
if os.path.exists(prediction_data_filename):
    # Append to existing file
    existing_data = pd.read_csv(prediction_data_filename)
    updated_data = pd.concat([existing_data, pd.DataFrame([prediction_record])], ignore_index=True)
    updated_data.to_csv(prediction_data_filename, index=False)
else:
    # Create new file
    pd.DataFrame([prediction_record]).to_csv(prediction_data_filename, index=False)

st.subheader("✅ Prediction Log")
st.success(f"Prediction record saved to {prediction_data_filename} at {prediction_timestamp}")
st.info(f"Current day prediction (top of table): {currency_symbol}{top_prediction:.2f}")

# Add accuracy information to the display
st.info(f"Error Rate: {current_day_error_rate:.2f}% | Accuracy: {current_day_accuracy:.2f}%")

# Download prediction log
with open(prediction_data_filename, 'rb') as file:
    st.download_button(
        label="Download Prediction Log",
        data=file,
        file_name=prediction_data_filename,
        mime='text/csv'
    )

# Add historical accuracy tracking if log file exists
if os.path.exists(prediction_data_filename):
    st.subheader("📊 Historical Prediction Accuracy")
    
    try:
        historical_data = pd.read_csv(prediction_data_filename)
        # Only show last 10 entries to keep the display clean
        recent_history = historical_data.tail(10)
        
        # Create accuracy trend chart
        if 'Accuracy_Percent' in recent_history.columns and len(recent_history) > 1:
            fig_accuracy = go.Figure()
            fig_accuracy.add_trace(go.Scatter(
                x=recent_history['Timestamp'], 
                y=recent_history['Accuracy_Percent'],
                mode='lines+markers',
                name='Prediction Accuracy',
                line=dict(color='green')
            ))
            fig_accuracy.update_layout(
                title="Prediction Accuracy Trend",
                xaxis_title="Timestamp",
                yaxis_title="Accuracy (%)",
                yaxis=dict(range=[0, 100])
            )
            st.plotly_chart(fig_accuracy)
            
            # Calculate average accuracy
            avg_accuracy = recent_history['Accuracy_Percent'].mean()
            st.write(f"Average prediction accuracy (last 10 predictions): *{avg_accuracy:.2f}%*")
    except Exception as e:
        st.error(f"Error displaying historical accuracy: {e}")

# Option to download historical prediction data
if not historical_df.empty:
    historical_csv = f"{stock}_historical_predictions.csv"
    historical_df.to_csv(historical_csv, index=False)
    with open(historical_csv, 'rb') as file:
        st.download_button(
            label="Download 30-Day Historical Predictions",
            data=file,
            file_name=historical_csv,
            mime='text/csv'
        )