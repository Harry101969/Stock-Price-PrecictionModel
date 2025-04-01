import numpy as np
import pandas as pd
import streamlit as st
import datetime as dt
import yfinance as yf
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('stockpricemodel.keras')

st.title("Stock Price Prediction App")

# User input for stock symbol
stock = st.text_input("Enter Stock Symbol (e.g., POWERGRID.NS)", "POWERGRID.NS")

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

# 🚀 Predict the next 5–6 years recursively (~180 days)
last_100_days = input_data[-100:]
future_predictions = []
current_input = last_100_days.reshape(1, 100, 1)

# Predict next 30 days
previous_price = y_test[-1]
for _ in range(30):  # 180 days
    next_prediction = model.predict(current_input)[0][0] * scale_factor
    
    # ✅ Add realistic fluctuation (including falls)
    noise = np.random.uniform(-5, 5)  # Simulate realistic fluctuation
    next_prediction += noise
    
    if len(future_predictions) > 0:
        next_prediction = np.clip(next_prediction, future_predictions[-1] - 2.4930, future_predictions[-1] + 3.0489)
    else:
        next_prediction = np.clip(next_prediction, previous_price - 2.5061, previous_price + 2.6137)
    
    future_predictions.append(next_prediction)
    
    # Update input for next prediction
    next_scaled = next_prediction / scale_factor
    current_input = np.append(current_input[:, 1:, :], [[[next_scaled]]], axis=1)

# Reverse scaling for target values
future_predictions = np.array(future_predictions)
future_dates = pd.date_range(start=end + dt.timedelta(days=1), periods=30)

# ✅ Table only for 5–6 months (~180 days)
target_df = pd.DataFrame({
    'Date': future_dates[:10].strftime('%Y-%m-%d'),
    'Predicted Target Price': future_predictions[:10]
})

# Display table for next 5–6 months only
st.subheader("Predicted Target for Next 5 Days")
st.write(target_df)

# Get last closing price correctly as float
current_price = df['Close'].iloc[-1]

import pandas as pd

# Ensure current_price is a float
current_price = float(current_price) if isinstance(current_price, (pd.Series, pd.DataFrame)) else current_price

ticker_info = yf.Ticker(stock).info
currency = ticker_info.get("currency", "INR") 
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

formatted_price = f"Current Stock Price: **{currency_symbol}{current_price}**"
st.write(formatted_price)

shares_bought = st.number_input("Enter the number of shares you want to buy:", min_value=1, step=1, value=10)

next_day_price = float(y_predicted[-1][0])  # Ensure it's also a float
profit_loss = (next_day_price - current_price) * shares_bought

if profit_loss > 0:
    st.success(f"🎉 Expected **Profit** for next day: **{currency_symbol}{profit_loss:.2f}**")
elif profit_loss < 0:
    st.error(f"⚠️ Expected **Loss** for next day: **{currency_symbol}{profit_loss:.2f}**")
else:
    st.info("No gain or loss expected for the next day.")


# ✅ Plot future trend for next 5–6 years
# st.subheader("Future Trend for Next 5–6 Years")
# fig4 = go.Figure()
# fig4.add_trace(go.Scatter(
#     x=future_dates, y=future_predictions, mode='lines', name="Predicted Price", line=dict(color="orange")
# ))

# fig4.update_layout(title="Future Stock Price Prediction (Next 5–6 Years)",
#                    xaxis_title="Date",
#                    yaxis_title="Price",
#                    xaxis_rangeslider_visible=False)
# st.plotly_chart(fig4)

# ✅ Download dataset
csv_file_path = f"{stock}_dataset.csv"
df.to_csv(csv_file_path)
st.download_button(label="Download Dataset as CSV", data=open(csv_file_path, 'rb'), file_name=csv_file_path, mime='text/csv')