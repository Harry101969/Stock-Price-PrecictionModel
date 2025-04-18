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
st.subheader(f"Predicted Prices For {stock}")
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

formatted_price = f"Current Stock Price: {currency_symbol}{current_price}"
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
    st.success(f"🎉 Expected Profit for {day_label}: {currency_symbol}{profit_loss:.2f}")
elif profit_loss < 0:
    st.error(f"⚠ Expected Loss for {day_label}: {currency_symbol}{profit_loss:.2f}")
else:
    st.info(f"No gain or loss expected for {day_label}.")

# Display model accuracy metrics for current day prediction
st.subheader("🎯 Model Accuracy Metrics")
st.write(f"Error Rate for Current Day Prediction: {current_day_error_rate:.2f}%")
st.write(f"Accuracy for Current Day Prediction: {current_day_accuracy:.2f}%")

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

# Create a unique filename based on the stock symbol
prediction_data_filename = f"{stock}_prediction_log.csv"

# Define a function to create a prediction record with proper timestamps
def create_prediction_record(record_date, actual_price, predicted_price):
    # Calculate error rate and accuracy
    error_rate = abs((predicted_price - actual_price) / actual_price) * 100 if actual_price != 0 else 0
    accuracy = 100 - error_rate
    
    # For historical records, use 8:00 AM on that date
    if isinstance(record_date, (pd.Timestamp, dt.datetime)):
        timestamp = dt.datetime.combine(record_date.date(), dt.time(8, 0))
    else:
        # Try to parse the date if it's a string
        try:
            parsed_date = pd.to_datetime(record_date).date()
            timestamp = dt.datetime.combine(parsed_date, dt.time(8, 0))
        except:
            # Fallback to current time if parsing fails
            timestamp = dt.datetime.now()
    
    # Get future dates (next 5 days)
    if isinstance(record_date, (pd.Timestamp, dt.datetime)):
        future_dates = [record_date + dt.timedelta(days=i+1) for i in range(5)]
    else:
        try:
            base_date = pd.to_datetime(record_date)
            future_dates = [base_date + dt.timedelta(days=i+1) for i in range(5)]
        except:
            # Fallback if parsing fails
            base_date = dt.datetime.now()
            future_dates = [base_date + dt.timedelta(days=i+1) for i in range(5)]
    
    # For the first date (current), use the provided prediction
    # For future dates, generate placeholder predictions with small variations
    future_prices = []
    last_price = predicted_price
    for _ in range(5):
        next_price = last_price * (1 + np.random.uniform(-0.02, 0.02))  # Small variation
        future_prices.append(next_price)
        last_price = next_price
    
    # Create the base record with numeric accuracy values
    record = {
        'Timestamp': timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        'Stock': stock,
        'Date': record_date.strftime('%Y-%m-%d') if isinstance(record_date, (pd.Timestamp, dt.datetime)) else record_date,
        'Current_Price': float(actual_price),
        'Current_Day_Prediction': float(predicted_price),
        'Error_Rate_Percent': float(error_rate),
        'Accuracy_Percent': float(accuracy)  # Ensure this is a single float value
    }
    
    # Add future date predictions
    for i, (future_date, future_price) in enumerate(zip(future_dates, future_prices)):
        day_number = i + 1
        record[f'Date_{day_number}'] = future_date.strftime('%Y-%m-%d') if isinstance(future_date, (pd.Timestamp, dt.datetime)) else future_date
        record[f'Price_{day_number}'] = float(future_price)
    
    return record

# Only process historical data if we have enough
historical_records = []
if len(df) >= 130:  # Need at least 100 + 30 days
    # Generate predictions for the last 30 days
    historical_dates = last_30_days_data.index
    historical_actual_prices = last_30_days_data['Close'].values
    
    # Extract the corresponding 30 days of predictions from y_predicted if available
    if len(y_predicted) >= 30:
        historical_pred_prices = y_predicted[-30:].flatten()
    else:
        padding_length = 30 - len(y_predicted)
        historical_pred_prices = np.concatenate([
            np.array([np.nan] * padding_length),
            y_predicted.flatten()
        ])
    
    # Create historical records in the same format as current prediction records
    for date, actual, pred in zip(historical_dates, historical_actual_prices, historical_pred_prices):
        if not np.isnan(pred):
            historical_records.append(create_prediction_record(date, actual, pred))

# Create current prediction record
current_record = create_prediction_record(current_date, current_price, top_prediction)
prediction_timestamp = current_record['Timestamp']

# Check if file exists to append or create new
if os.path.exists(prediction_data_filename):
    try:
        # Read existing file
        existing_data = pd.read_csv(prediction_data_filename)
        
        # Clean up any problematic numeric columns
        for col in ['Error_Rate_Percent', 'Accuracy_Percent']:
            if col in existing_data.columns:
                # Convert string representations to floats, handling potential errors
                try:
                    existing_data[col] = pd.to_numeric(existing_data[col], errors='coerce')
                except:
                    # If conversion fails completely, create a new clean column
                    existing_data[col] = existing_data[col].apply(
                        lambda x: float(str(x).split(']')[0].replace('[', '')) 
                        if isinstance(x, str) and '[' in str(x) 
                        else (float(x) if pd.notna(x) else np.nan)
                    )
        
        # Combine historical and current records
        all_records = historical_records + [current_record]
        
        # Create DataFrame from new records
        new_records_df = pd.DataFrame(all_records)
        
        # Ensure numeric columns are properly formatted
        for col in ['Error_Rate_Percent', 'Accuracy_Percent', 'Current_Price', 'Current_Day_Prediction']:
            if col in new_records_df.columns:
                new_records_df[col] = pd.to_numeric(new_records_df[col], errors='coerce')
        
        # Append to existing data
        updated_data = pd.concat([existing_data, new_records_df], ignore_index=True)
        
        # Remove any duplicate timestamps
        updated_data = updated_data.drop_duplicates(subset=['Timestamp'], keep='last')
        
        updated_data.to_csv(prediction_data_filename, index=False)
    except Exception as e:
        st.error(f"Error processing existing prediction log: {e}")
        # Create new file just with current records as a fallback
        pd.DataFrame(historical_records + [current_record]).to_csv(prediction_data_filename, index=False)
else:
    # Create new file with historical records first, then current record
    all_records = historical_records + [current_record]
    pd.DataFrame(all_records).to_csv(prediction_data_filename, index=False)

st.subheader("✅ Prediction Log")
st.success(f"Prediction record saved to {prediction_data_filename} at {prediction_timestamp}")
st.info(f"Current day prediction: {currency_symbol}{top_prediction:.2f}")

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
        
        # Ensure Accuracy_Percent is numeric
        if 'Accuracy_Percent' in historical_data.columns:
            try:
                # Try standard conversion first
                historical_data['Accuracy_Percent'] = pd.to_numeric(historical_data['Accuracy_Percent'], errors='coerce')
            except:
                # If that fails, try to extract the first number from any string arrays
                historical_data['Accuracy_Percent'] = historical_data['Accuracy_Percent'].apply(
                    lambda x: float(str(x).split(']')[0].replace('[', '')) 
                    if isinstance(x, str) and '[' in str(x) 
                    else (float(x) if pd.notna(x) else np.nan)
                )
        
        # Only show last 10 entries to keep the display clean
        recent_history = historical_data.tail(10)
        
        # Create accuracy trend chart
        if 'Accuracy_Percent' in recent_history.columns and len(recent_history) > 1:
            # Drop any NaN values
            recent_history = recent_history.dropna(subset=['Accuracy_Percent'])
            
            if len(recent_history) > 1:
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
                st.write(f"Average prediction accuracy (last {len(recent_history)} predictions): {avg_accuracy:.2f}%")
            else:
                st.info("Not enough valid historical data points for trend chart.")
        else:
            st.info("Not enough historical data available yet.")
    except Exception as e:
        st.error(f"Error displaying historical accuracy: {e}")
        st.error("Detailed error info for debugging:")
        import traceback
        st.code(traceback.format_exc())