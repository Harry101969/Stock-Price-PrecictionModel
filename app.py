# # import numpy as np
# # import pandas as pd
# # import matplotlib.pyplot as plt
# # # from tensorflow.keras.models import load_model
# # from keras.models import load_model  # type: ignore
 
# # import streamlit as st
# # import datetime as dt
# # import yfinance as yf
# # from sklearn.preprocessing import MinMaxScaler

# # # Ensure TensorFlow is installed
# # try:
# #     import tensorflow as tf
# # except ImportError:
# #     st.error("TensorFlow is not installed. Please install it using: pip install tensorflow")

# # # Load the model
# # try:
# #     model = load_model('stock_detection_model.h5')
# # except Exception as e:
# #     st.error(f"Error loading model: {e}")

# # # Streamlit UI
# # def main():
# #     st.title("📈 Stock Price Prediction App")
    
# #     stock = st.text_input("Enter Stock Ticker:", "POWERGRID.NS")
# #     start = dt.datetime(2000, 1, 1)
# #     end = dt.datetime(2024, 10, 1)
    
# #     if st.button("Predict"):  
# #         try:
# #             df = yf.download(stock, start=start, end=end)
# #             if df.empty:
# #                 st.error("Invalid stock ticker or no data available.")
# #                 return
            
# #             st.subheader("Stock Data Preview")
# #             st.write(df.tail())
            
# #             # Exponential Moving Averages
# #             ema20 = df.Close.ewm(span=20, adjust=False).mean()
# #             ema50 = df.Close.ewm(span=50, adjust=False).mean()
# #             ema100 = df.Close.ewm(span=100, adjust=False).mean()
# #             ema200 = df.Close.ewm(span=200, adjust=False).mean()
            
# #             # Data processing
# #             data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
# #             data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])
# #             scaler = MinMaxScaler(feature_range=(0, 1))
# #             data_training_array = scaler.fit_transform(data_training)
            
# #             past_100_days = data_training.tail(100)
# #             final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
# #             input_data = scaler.fit_transform(final_df)
            
# #             x_test, y_test = [], []
# #             for i in range(100, input_data.shape[0]):
# #                 x_test.append(input_data[i - 100:i])
# #                 y_test.append(input_data[i, 0])
# #             x_test, y_test = np.array(x_test), np.array(y_test)
            
# #             # Predict
# #             y_predicted = model.predict(x_test)
            
# #             # Inverse scaling
# #             scaler = scaler.scale_
# #             scale_factor = 1 / scaler[0]
# #             y_predicted = y_predicted * scale_factor
# #             y_test = y_test * scale_factor
            
# #             # Plot EMA Chart
# #             st.subheader("Closing Price with EMA (20 & 50 Days)")
# #             fig1, ax1 = plt.subplots(figsize=(12, 6))
# #             ax1.plot(df.Close, 'y', label='Closing Price')
# #             ax1.plot(ema20, 'g', label='EMA 20')
# #             ax1.plot(ema50, 'r', label='EMA 50')
# #             ax1.set_xlabel("Time")
# #             ax1.set_ylabel("Price")
# #             ax1.legend()
# #             st.pyplot(fig1)
            
# #             # Plot Prediction vs Original
# #             st.subheader("Prediction vs Original Trend")
# #             fig2, ax2 = plt.subplots(figsize=(12, 6))
# #             ax2.plot(y_test, 'g', label="Original Price")
# #             ax2.plot(y_predicted, 'r', label="Predicted Price")
# #             ax2.set_xlabel("Time")
# #             ax2.set_ylabel("Price")
# #             ax2.legend()
# #             st.pyplot(fig2)
            
# #             # Download CSV
# #             csv_file_path = f"{stock}_dataset.csv"
# #             df.to_csv(csv_file_path)
# #             with open(csv_file_path, "rb") as f:
# #                 st.download_button("Download CSV", f, file_name=csv_file_path)
                
# #         except Exception as e:
# #             st.error(f"An error occurred: {e}")
        
# # if __name__ == "__main__":
# #     main()

# import numpy as np
# import pandas as pd
# import plotly.graph_objects as go
# import pickle
# import streamlit as st
# import datetime as dt
# import yfinance as yf
# from sklearn.preprocessing import MinMaxScaler

# # Load the model from .pkl file
# try:
#     with open("stock_detection_model.pkl", "rb") as f:  # Updated pkl filename
#         model = pickle.load(f)
# except Exception as e:
#     st.error(f"Error loading model: {e}")

# # Streamlit UI
# def main():
#     st.title("📈 Stock Price Prediction App")

#     stock = st.text_input("Enter Stock Ticker:", "POWERGRID.NS")
#     start = dt.datetime(2000, 1, 1)
#     end = dt.datetime(2025, 2, 27)  # Updated end date to Feb 27, 2025

#     if st.button("Predict"):
#         try:
#             df = yf.download(stock, start=start, end=end)
#             if df.empty:
#                 st.error("Invalid stock ticker or no data available.")
#                 return

#             st.subheader("Stock Data Preview")
#             st.write(df.tail())

#             # Exponential Moving Averages
#             ema20 = df.Close.ewm(span=20, adjust=False).mean()
#             ema50 = df.Close.ewm(span=50, adjust=False).mean()

#             # Data processing
#             data_training = df['Close'][0:int(len(df) * 0.70)]
#             data_testing = df['Close'][int(len(df) * 0.70):]
#             scaler = MinMaxScaler(feature_range=(0, 1))
#             data_training_array = scaler.fit_transform(data_training.values.reshape(-1, 1))

#             past_100_days = data_training[-100:]
#             final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
#             input_data = scaler.fit_transform(final_df.values.reshape(-1, 1))

#             x_test, y_test = [], []
#             for i in range(100, input_data.shape[0]):
#                 x_test.append(input_data[i - 100:i])
#                 y_test.append(input_data[i, 0])
#             x_test, y_test = np.array(x_test), np.array(y_test)

#             # Predict using the loaded model
#             y_predicted = model.predict(x_test)

#             # Inverse scaling
#             scale_factor = 1 / scaler.scale_[0]
#             y_predicted = y_predicted * scale_factor
#             y_test = y_test * scale_factor

#             # Extract corresponding dates for the test dataset
#             test_dates = df.index[-len(y_test):]

#             # 📌 Interactive Candlestick Chart
#             st.subheader("📊 Interactive Candlestick Chart with EMA")
#             fig1 = go.Figure()

#             # Candlestick data
#             fig1.add_trace(go.Candlestick(
#                 x=df.index,
#                 open=df['Open'],
#                 high=df['High'],
#                 low=df['Low'],
#                 close=df['Close'],
#                 name="Candlestick"
#             ))

#             # Exponential Moving Averages
#             fig1.add_trace(go.Scatter(x=df.index, y=ema20, mode="lines", name="EMA 20", line=dict(color="green")))
#             fig1.add_trace(go.Scatter(x=df.index, y=ema50, mode="lines", name="EMA 50", line=dict(color="red")))

#             fig1.update_layout(
#                 xaxis_rangeslider_visible=False,
#                 title="Stock Price Chart with Moving Averages",
#                 xaxis_title="Date",
#                 yaxis_title="Price",
#                 template="plotly_dark"
#             )
#             st.plotly_chart(fig1)
# #             # Creating the candlestick 
# # import plotly.graph_objects as go
# # fig = go.Figure(data=[go.Candlestick(x=data1['Date'],
# #                                      open=data1['Open'],
# #                                      high=data1['High'],
# #                                      low=data1['Low'],
# #                                      close=data1['Close'])])
# # fig.update_layout(xaxis_rangeslider_visible=False)
# # fig.show()

#             # 📌 Interactive Prediction vs Original Trend
#             st.subheader("📈 Prediction vs Original Trend")
#             fig2 = go.Figure()

#             fig2.add_trace(go.Scatter(x=test_dates, y=y_test, mode="lines", name="Original Price", line=dict(color="green")))
#             fig2.add_trace(go.Scatter(x=test_dates, y=y_predicted.flatten(), mode="lines", name="Predicted Price", line=dict(color="red")))

#             fig2.update_layout(
#                 title="Original vs Predicted Stock Price",
#                 xaxis_title="Date",
#                 yaxis_title="Price",
#                 template="plotly_dark"
#             )
#             st.plotly_chart(fig2)

#             # Download CSV
#             csv_file_path = f"{stock}_dataset.csv"
#             df.to_csv(csv_file_path)
#             with open(csv_file_path, "rb") as f:
#                 st.download_button("Download CSV", f, file_name=csv_file_path)

#         except Exception as e:
#             st.error(f"An error occurred: {e}")

# if __name__ == "__main__":
#     main()

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pickle
import streamlit as st
import datetime as dt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

# Load the model from .pkl file
try:
    with open("stock_detection_model.pkl", "rb") as f:  # Updated pkl filename
        model = pickle.load(f)
except Exception as e:
    st.error(f"Error loading model: {e}")

# Streamlit UI
def main():
    st.title("📈 Stock Price Prediction App")

    stock = st.text_input("Enter Stock Ticker:", "POWERGRID.NS")
    start = dt.datetime(2000, 1, 1)
    end = dt.datetime(2025, 2, 27)  # Updated end date to Feb 27, 2025

    if st.button("Predict"):
        try:
            df = yf.download(stock, start=start, end=end)
            if df.empty:
                st.error("Invalid stock ticker or no data available.")
                return

            st.subheader("Stock Data Preview")
            st.write(df.tail())

            # Exponential Moving Averages (Corrected)
            df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
            df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()

            # Data processing
            data_training = df['Close'][0:int(len(df) * 0.70)]
            data_testing = df['Close'][int(len(df) * 0.70):]
            scaler = MinMaxScaler(feature_range=(0, 1))
            data_training_array = scaler.fit_transform(data_training.values.reshape(-1, 1))

            past_100_days = data_training[-100:]
            final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
            input_data = scaler.fit_transform(final_df.values.reshape(-1, 1))

            x_test, y_test = [], []
            for i in range(100, input_data.shape[0]):
                x_test.append(input_data[i - 100:i])
                y_test.append(input_data[i, 0])
            x_test, y_test = np.array(x_test), np.array(y_test)

            # Predict using the loaded model
            y_predicted = model.predict(x_test)

            # Inverse scaling
            scale_factor = 1 / scaler.scale_[0]
            y_predicted = y_predicted * scale_factor
            y_test = y_test * scale_factor

            # Extract corresponding dates for the test dataset
            test_dates = df.index[-len(y_test):]

            # 📌 **Fixed Interactive EMA Chart**
            st.subheader("📊 Interactive Stock Chart with EMA")
            fig1 = go.Figure()

            # Candlestick Chart
            fig1.add_trace(go.Candlestick(
                x=df.index, open=df['Open'], high=df['High'],
                low=df['Low'], close=df['Close'], name='Candlestick'
            ))

            # EMA Lines
            fig1.add_trace(go.Scatter(
                x=df.index, y=df['EMA_20'], mode='lines', name="EMA 20", line=dict(color="green")
            ))
            fig1.add_trace(go.Scatter(
                x=df.index, y=df['EMA_50'], mode='lines', name="EMA 50", line=dict(color="red")
            ))

            # Layout Settings
            fig1.update_layout(title="Stock Prices with EMA",
                               xaxis_title="Date",
                               yaxis_title="Price",
                               xaxis_rangeslider_visible=False)
            st.plotly_chart(fig1)

            # 📊 **Prediction vs Original Price Chart**
            st.subheader("📈 Prediction vs Original Price")
            fig2 = go.Figure()

            fig2.add_trace(go.Scatter(
                x=test_dates, y=y_test, mode='lines', name="Actual Price", line=dict(color="blue")
            ))
            fig2.add_trace(go.Scatter(
                x=test_dates, y=y_predicted.flatten(), mode='lines', name="Predicted Price", line=dict(color="orange")
            ))

            fig2.update_layout(title="Stock Price Prediction",
                               xaxis_title="Date",
                               yaxis_title="Price",
                               xaxis_rangeslider_visible=False)
            st.plotly_chart(fig2)

            # 📥 **Download CSV**
            csv_file_path = f"{stock}_dataset.csv"
            df.to_csv(csv_file_path)
            with open(csv_file_path, "rb") as f:
                st.download_button("Download CSV", f, file_name=csv_file_path)

        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
