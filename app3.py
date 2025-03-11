# # import streamlit as st
# # import pandas as pd
# # import numpy as np
# # import datetime as dt
# # import yfinance as yf
# # from tensorflow.keras.models import load_model
# # from sklearn.preprocessing import MinMaxScaler
# # import plotly.graph_objects as go

# # # Load the trained model
# # model = load_model('stockpricemodel.keras')

# # # Title
# # st.title("Stock Price Prediction App")

# # # User input for stock symbol
# # stock_symbol = st.text_input("Enter Stock Symbol (e.g., AAPL):")

# # # Function to convert string to datetime
# # def str_to_datetime(s):
# #     year, month, day = map(int, s.split('-'))
# #     return dt.datetime(year, month, day)

# # # Function to fetch and process stock data
# # def get_stock_data(stock_symbol):
# #     try:
# #         start = dt.datetime(2000, 1, 1)
# #         end = dt.datetime.now()
# #         df = yf.download(stock_symbol, start, end)
        
# #         # Reset index and drop unwanted row/column
# #         df.reset_index(inplace=True)
# #         df = df[['Date', 'Close']]
# #         df['Date'] = df['Date'].astype(str).apply(str_to_datetime)
# #         df.set_index('Date', inplace=True)

# #         return df
# #     except Exception as e:
# #         st.error(f"Error fetching stock data: {e}")
# #         return None

# # # Function to prepare data for prediction
# # def df_to_windowed_df(dataframe, first_date_str, last_date_str, n=3):
# #     first_date = str_to_datetime(first_date_str)
# #     last_date  = str_to_datetime(last_date_str)

# #     target_date = first_date
# #     dates = []
# #     X, Y = [], []

# #     last_time = False
# #     while True:
# #         df_subset = dataframe.loc[:target_date].tail(n + 1)
        
# #         if len(df_subset) != n + 1:
# #             target_date += dt.timedelta(days=1)
# #             continue

# #         values = df_subset['Close'].to_numpy()
# #         x, y = values[:-1], values[-1]

# #         dates.append(target_date)
# #         X.append(x)
# #         Y.append(y)

# #         next_week = dataframe.loc[target_date:target_date + dt.timedelta(days=7)]
# #         if len(next_week) < 2:
# #             break

# #         next_datetime_str = str(next_week.head(2).tail(1).index.values[0])
# #         next_date_str = next_datetime_str.split('T')[0]
# #         year, month, day = map(int, next_date_str.split('-'))
# #         next_date = dt.datetime(year=year, month=month, day=day)

# #         if last_time:
# #             break
        
# #         target_date = next_date

# #         if target_date == last_date:
# #             last_time = True
    
# #     ret_df = pd.DataFrame({'Target Date': dates})
    
# #     X = np.array(X)
# #     for i in range(n):
# #         ret_df[f'Target-{n - i}'] = X[:, i]
    
# #     ret_df['Target'] = Y

# #     return ret_df

# # # Prediction button
# # if st.button("Predict"):
# #     if stock_symbol:
# #         df = get_stock_data(stock_symbol)
# #         if df is not None:
# #             # Split data for training and testing
# #             data_training = df[['Close']][:int(len(df) * 0.70)]
# #             data_testing = df[['Close']][int(len(df) * 0.70):]

# #             scaler = MinMaxScaler(feature_range=(0, 1))
# #             data_training_array = scaler.fit_transform(data_training)

# #             # Prepare test data
# #             past_100_days = data_training.tail(100)
# #             final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
# #             input_data = scaler.transform(final_df)

# #             x_test, y_test = [], []
# #             for i in range(100, input_data.shape[0]):
# #                 x_test.append(input_data[i - 100:i])
# #                 y_test.append(input_data[i, 0])
# #             x_test, y_test = np.array(x_test), np.array(y_test)

# #             # Predict using the model
# #             y_predicted = model.predict(x_test)

# #             # Reverse scaling
# #             scale_factor = 1 / scaler.scale_[0]
# #             y_predicted = y_predicted * scale_factor
# #             y_test = y_test * scale_factor

# #             # Prepare windowed data for output table
# #             first_valid_date = df.iloc[3].name
# #             windowed_df = df_to_windowed_df(
# #                 df,
# #                 str(first_valid_date.date()),
# #                 str(df.index[-1].date()),
# #                 n=3
# #             )

# #             # Display the output table
# #             st.subheader("Predicted Targets")
# #             st.table(windowed_df)

# #             # Extract corresponding dates for plotting
# #             test_dates = df.index[-len(y_test):]

# #             # Plot Predictions vs Actual Data
# #             st.subheader("Prediction vs Actual Data")
# #             fig = go.Figure()
# #             fig.add_trace(go.Scatter(
# #                 x=test_dates, y=y_test, mode='lines',
# #                 name="Actual Price", line=dict(color="blue")
# #             ))
# #             fig.add_trace(go.Scatter(
# #                 x=test_dates, y=y_predicted.flatten(), mode='lines',
# #                 name="Predicted Price", line=dict(color="orange")
# #             ))
# #             fig.update_layout(
# #                 title="Stock Price Prediction",
# #                 xaxis_title="Date",
# #                 yaxis_title="Price",
# #                 xaxis_rangeslider_visible=False
# #             )
# #             st.plotly_chart(fig)

# #         else:
# #             st.error("Failed to fetch data.")
# #     else:
# #         st.warning("Please enter a stock symbol.")

# # # # # Optional Download Button
# # # # # csv_file_path = f"{stock_symbol}_dataset.csv"
# # # # # df.to_csv(csv_file_path)
# # # # # st.download_button(
# # # # #     label="Download Dataset as CSV",
# # # # #     data=open(csv_file_path, 'rb'),
# # # # #     file_name=csv_file_path,
# # # # #     mime='text/csv'
# # # # # )

# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import yfinance as yf
# import datetime as dt
# import plotly.graph_objects as go
# from tensorflow.keras.models import load_model
# from sklearn.preprocessing import MinMaxScaler
# from copy import deepcopy

# # Load the trained model
# model = load_model('stockpricemodel.keras')

# st.title("Stock Price Prediction App")

# # User input for stock symbol
# stock = st.text_input("Enter Stock Symbol (e.g., POWERGRID.NS)", "POWERGRID.NS")

# # Function to fetch stock data
# def get_stock_data(stock_symbol):
#     start = dt.datetime(2000, 1, 1)
#     end = dt.datetime.now()
#     df = yf.download(stock_symbol, start=start, end=end)

#     if not df.empty:
#         df = df.reset_index()
#         df = df[['Date', 'Close']]
#         df['Date'] = pd.to_datetime(df['Date'])
#         df.set_index('Date', inplace=True)
#         return df
#     else:
#         st.error("Failed to fetch data. Please check the stock symbol.")
#         return None

# # Prepare the data for prediction
# def prepare_data(df):
#     data_training = df[['Close']][:int(len(df) * 0.90)]
#     data_testing = df[['Close']][int(len(df) * 0.80):]

#     scaler = MinMaxScaler(feature_range=(0, 1))
#     data_training_array = scaler.fit_transform(data_training)

#     past_100_days = data_training.tail(100)
#     final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
#     input_data = scaler.transform(final_df)

#     x_test, y_test = [], []
#     for i in range(100, input_data.shape[0]):
#         x_test.append(input_data[i - 100:i])
#         y_test.append(input_data[i, 0])

#     x_test, y_test = np.array(x_test), np.array(y_test)
    
#     return x_test, y_test, scaler

# # Predict future prices
# def predict_future(df, scaler):
#     last_window = deepcopy(df[['Close']].tail(100).values)
#     last_window = scaler.transform(last_window)

#     future_dates = pd.date_range(start=df.index[-1], periods=6, freq='B')[1:]

#     predictions = []
#     for date in future_dates:
#         prediction = model.predict(np.array([last_window])).flatten()[0]
#         predictions.append(prediction)
#         last_window = np.roll(last_window, -1)
#         last_window[-1] = prediction

#     predictions = np.array(predictions) * (1 / scaler.scale_[0])

#     return future_dates, predictions

# # Prepare windowed data
# def df_to_windowed_df(dataframe, n=3):
#     dates = []
#     X, Y = [], []

#     for i in range(n, len(dataframe)):
#         x = dataframe['Close'].iloc[i-n:i].values
#         y = dataframe['Close'].iloc[i]
#         dates.append(dataframe.index[i])
#         X.append(x)
#         Y.append(y)

#     X = np.array(X).reshape(-1, n, 1)
#     Y = np.array(Y)

#     return dates, X, Y

# # Prediction button
# if st.button("Predict"):
#     if stock:
#         df = get_stock_data(stock)
#         if df is not None:
#             # Prepare test data
#             x_test, y_test, scaler = prepare_data(df)

#             # Prepare windowed data
#             dates, X, y = df_to_windowed_df(df)

#             # Train/validation/test split
#             q_80 = int(len(dates) * 0.8)
#             q_90 = int(len(dates) * 0.9)

#             dates_train, X_train, y_train = dates[:q_80], X[:q_80], y[:q_80]
#             dates_val, X_val, y_val = dates[q_80:q_90], X[q_80:q_90], y[q_80:q_90]
#             dates_test, X_test, y_test = dates[q_90:], X[q_90:], y[q_90:]

#             # Predict for train, validation, and test sets
#             train_predictions = model.predict(X_train).flatten()
#             val_predictions = model.predict(X_val).flatten()
#             test_predictions = model.predict(X_test).flatten()

#             # Make predictions for actual test data
#             y_predicted = model.predict(x_test).flatten()
#             y_predicted = y_predicted * (1 / scaler.scale_[0])
#             y_test = y_test * (1 / scaler.scale_[0])
#             test_dates = df.index[-len(y_test):]

#             # Plot Actual vs Predicted graph
#             st.subheader("Prediction vs Original Trend")
#             fig1 = go.Figure()
#             fig1.add_trace(go.Scatter(
#                 x=test_dates, y=y_test, mode='lines', name="Actual Price", line=dict(color="blue")
#             ))
#             fig1.add_trace(go.Scatter(
#                 x=test_dates, y=y_predicted, mode='lines', name="Predicted Price", line=dict(color="orange")
#             ))
#             fig1.update_layout(
#                 title="Stock Price Prediction",
#                 xaxis_title="Date",
#                 yaxis_title="Price",
#                 xaxis_rangeslider_visible=False
#             )
#             st.plotly_chart(fig1)

#             # Predict future target values
#             future_dates, future_preds = predict_future(df, scaler)

#             # Display target table
#             target_table = pd.DataFrame({
#                 'Target Date': future_dates.strftime('%Y-%m-%d'),
#                 'Target': future_preds
#             })
#             st.subheader("Next 5-Day Target Prices")
#             st.table(target_table)

#             # Plot Recursive Predictions graph
#             st.subheader("Recursive Predictions for Next 5 Days")
#             fig2 = go.Figure()
#             fig2.add_trace(go.Scatter(
#                 x=dates_train, y=train_predictions, mode='lines', name='Train Prediction', line=dict(color="orange")
#             ))
#             fig2.add_trace(go.Scatter(
#                 x=dates_val, y=val_predictions, mode='lines', name='Validation Prediction', line=dict(color="red")
#             ))
#             fig2.add_trace(go.Scatter(
#                 x=dates_test, y=test_predictions, mode='lines', name='Test Prediction', line=dict(color="pink")
#             ))
#             fig2.add_trace(go.Scatter(
#                 x=future_dates, y=future_preds, mode='lines', name='Recursive Prediction', line=dict(color="gold")
#             ))

#             fig2.update_layout(
#                 title="Recursive Predictions",
#                 xaxis_title="Date",
#                 yaxis_title="Price"
#             )
#             st.plotly_chart(fig2)

#     else:
#         st.warning("Please enter a stock symbol.")





# # import streamlit as st
# # import pandas as pd
# # import numpy as np
# # import datetime as dt
# # import plotly.graph_objects as go
# # from tensorflow.keras.models import Sequential, load_model
# # from tensorflow.keras.optimizers import Adam
# # from tensorflow.keras import layers
# # from copy import deepcopy
# # import yfinance as yf

# # # Load the trained model
# # model = load_model('stockpricemodel.keras')

# # # Function to fetch stock data
# # def get_stock_data(stock_symbol):
# #     try:
# #         start = dt.datetime(2000, 1, 1)
# #         end = dt.datetime.now()
# #         df = yf.download(stock_symbol, start=start, end=end)
# #         df.reset_index(inplace=True)
# #         df = df[['Date', 'Close']]
# #         df['Date'] = pd.to_datetime(df['Date'])
# #         df.set_index('Date', inplace=True)
# #         return df
# #     except Exception as e:
# #         st.error(f"Error fetching stock data: {e}")
# #         return None

# # # Function to prepare windowed data
# # def df_to_windowed_df(dataframe, n=3):
# #     dates = []
# #     X, Y = [], []

# #     for i in range(n, len(dataframe)):
# #         x = dataframe['Close'].iloc[i-n:i].values
# #         y = dataframe['Close'].iloc[i]
# #         dates.append(dataframe.index[i])
# #         X.append(x)
# #         Y.append(y)

# #     X = np.array(X).reshape(-1, n, 1)
# #     Y = np.array(Y)

# #     return dates, X, Y

# # # Function to predict next 5 days
# # def predict_next_5_days(dataframe):
# #     last_window = dataframe['Close'].values[-3:]
# #     predictions = []
# #     target_dates = pd.date_range(start=dataframe.index[-1] + pd.Timedelta(days=1), periods=5)

# #     for _ in range(5):
# #         input_data = last_window.reshape((1, 3, 1))
# #         next_pred = model.predict(input_data).flatten()[0]
# #         predictions.append(next_pred)

# #         # Shift window for next prediction
# #         last_window = np.append(last_window[1:], next_pred)

# #     return target_dates, predictions

# # # Title
# # st.title("Stock Price Prediction")

# # # User input
# # stock_symbol = st.text_input("Enter Stock Symbol (e.g., AAPL):", "AAPL")

# # # Prediction button
# # if st.button("Predict"):
# #     if stock_symbol:
# #         df = get_stock_data(stock_symbol)

# #         if df is not None:
# #             # Prepare windowed data
# #             dates, X, y = df_to_windowed_df(df)

# #             # Train/validation/test split
# #             q_80 = int(len(dates) * 0.8)
# #             q_90 = int(len(dates) * 0.9)

# #             dates_train, X_train, y_train = dates[:q_80], X[:q_80], y[:q_80]
# #             dates_val, X_val, y_val = dates[q_80:q_90], X[q_80:q_90], y[q_80:q_90]
# #             dates_test, X_test, y_test = dates[q_90:], X[q_90:], y[q_90:]

# #             # Predict for train, validation, and test sets
# #             train_predictions = model.predict(X_train).flatten()
# #             val_predictions = model.predict(X_val).flatten()
# #             test_predictions = model.predict(X_test).flatten()

# #             # Predict next 5 days using recursive prediction
# #             future_dates, future_preds = predict_next_5_days(df)

# #             # Display next 5 days prediction table
# #             result_df = pd.DataFrame({
# #                 'Date': future_dates.strftime('%Y-%m-%d'),
# #                 'Target': future_preds
# #             })
# #             st.subheader("Next 5 Days Target Price")
# #             st.table(result_df)

# #             # Plot Actual vs Predicted graph
# #             st.subheader("Actual vs Predicted Price")
# #             fig1 = go.Figure()
# #             fig1.add_trace(go.Scatter(x=dates_train, y=y_train, mode='lines', name='Train Data', line=dict(color="blue")))
# #             fig1.add_trace(go.Scatter(x=dates_train, y=train_predictions, mode='lines', name='Train Prediction', line=dict(color="orange")))
# #             fig1.add_trace(go.Scatter(x=dates_val, y=y_val, mode='lines', name='Validation Data', line=dict(color="green")))
# #             fig1.add_trace(go.Scatter(x=dates_val, y=val_predictions, mode='lines', name='Validation Prediction', line=dict(color="red")))
# #             fig1.add_trace(go.Scatter(x=dates_test, y=y_test, mode='lines', name='Test Data', line=dict(color="purple")))
# #             fig1.add_trace(go.Scatter(x=dates_test, y=test_predictions, mode='lines', name='Test Prediction', line=dict(color="pink")))

# #             fig1.update_layout(title="Actual vs Predicted Trend", xaxis_title="Date", yaxis_title="Price")
# #             st.plotly_chart(fig1)

# #             # Plot Recursive Predictions graph
# #             st.subheader("Recursive Predictions for Next 5 Days")
# #             fig2 = go.Figure()
# #             fig2.add_trace(go.Scatter(x=dates_train, y=train_predictions, mode='lines', name='Train Prediction', line=dict(color="orange")))
# #             fig2.add_trace(go.Scatter(x=dates_val, y=val_predictions, mode='lines', name='Validation Prediction', line=dict(color="red")))
# #             fig2.add_trace(go.Scatter(x=dates_test, y=test_predictions, mode='lines', name='Test Prediction', line=dict(color="pink")))
# #             fig2.add_trace(go.Scatter(x=future_dates, y=future_preds, mode='lines', name='Recursive Prediction', line=dict(color="gold")))

# #             fig2.update_layout(title="Recursive Predictions", xaxis_title="Date", yaxis_title="Price")
# #             st.plotly_chart(fig2)
# #         else:
# #             st.error("Failed to fetch data.")
# #     else:
# #         st.warning("Please enter a stock symbol.")

# # # Run using:
# # # streamlit run app.py


import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import datetime as dt
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from copy import deepcopy

# Load the trained model
model = load_model('stockpricemodel.keras')

st.title("Stock Price Prediction App")

# User input for stock symbol
stock = st.text_input("Enter Stock Symbol (e.g., POWERGRID.NS)", "POWERGRID.NS")

# Function to fetch stock data
def get_stock_data(stock_symbol):
    start = dt.datetime(2000, 1, 1)
    end = dt.datetime.now()
    df = yf.download(stock_symbol, start=start, end=end)

    if not df.empty:
        df = df.reset_index()
        df = df[['Date', 'Close']]
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        return df
    else:
        st.error("Failed to fetch data. Please check the stock symbol.")
        return None

# Prepare data for prediction
def prepare_data(df):
    data_training = df[['Close']][:int(len(df) * 0.70)]
    data_testing = df[['Close']][int(len(df) * 0.70):]

    scaler = MinMaxScaler(feature_range=(0, 1))
    data_training_array = scaler.fit_transform(data_training)

    past_100_days = data_training.tail(100)
    final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
    input_data = scaler.transform(final_df)

    x_test, y_test = [], []
    for i in range(100, input_data.shape[0]):
        x_test.append(input_data[i - 100:i])
        y_test.append(input_data[i, 0])

    x_test, y_test = np.array(x_test), np.array(y_test)
    
    return x_test, y_test, scaler

# Predict future prices (recursive)
def predict_future(df, scaler):
    last_window = deepcopy(df[['Close']].tail(100).values)
    last_window = scaler.transform(last_window)

    future_dates = pd.date_range(start=df.index[-1], periods=6, freq='B')[1:]

    predictions = []
    for date in future_dates:
        prediction = model.predict(np.array([last_window])).flatten()[0]
        prediction = prediction * (1 / scaler.scale_[0])

        # Ensure the difference is within a reasonable range (within ₹10/$10)
        if len(predictions) > 0:
            diff = abs(prediction - predictions[-1])
            if diff > 10:
                prediction = predictions[-1] + np.sign(prediction - predictions[-1]) * 10
        
        predictions.append(prediction)
        last_window = np.roll(last_window, -1)
        last_window[-1] = prediction * scaler.scale_[0]

    return future_dates, predictions

# Prepare windowed data for train/val/test split
def df_to_windowed_df(dataframe, n=3):
    dates = []
    X, Y = [], []

    for i in range(n, len(dataframe)):
        x = dataframe['Close'].iloc[i-n:i].values
        y = dataframe['Close'].iloc[i]
        dates.append(dataframe.index[i])
        X.append(x)
        Y.append(y)

    X = np.array(X).reshape(-1, n, 1)
    Y = np.array(Y)

    return dates, X, Y

# Prediction button
if st.button("Predict"):
    if stock:
        df = get_stock_data(stock)
        if df is not None:
            x_test, y_test, scaler = prepare_data(df)

            # Prepare windowed data
            dates, X, y = df_to_windowed_df(df)

            # Train/validation/test split
            q_80 = int(len(dates) * 0.8)
            q_90 = int(len(dates) * 0.9)

            dates_train, X_train, y_train = dates[:q_80], X[:q_80], y[:q_80]
            dates_val, X_val, y_val = dates[q_80:q_90], X[q_80:q_90], y[q_80:q_90]
            dates_test, X_test, y_test = dates[q_90:], X[q_90:], y[q_90:]

            # Predict for train, validation, and test sets
            train_predictions = model.predict(X_train).flatten()
            val_predictions = model.predict(X_val).flatten()
            test_predictions = model.predict(X_test).flatten()

            # Plot Actual vs Predicted
            st.subheader("Prediction vs Original Trend")
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(
                x=dates_test, y=y_test, mode='lines', name="Actual Price", line=dict(color="blue")
            ))
            fig1.add_trace(go.Scatter(
                x=dates_test, y=test_predictions, mode='lines', name="Predicted Price", line=dict(color="orange")
            ))
            fig1.update_layout(
                title="Stock Price Prediction",
                xaxis_title="Date",
                yaxis_title="Price",
                xaxis_rangeslider_visible=False
            )
            st.plotly_chart(fig1)

            # Predict future target values
            future_dates, future_targets = predict_future(df, scaler)

            # Display target table
            target_table = pd.DataFrame({
                'Target Date': future_dates.strftime('%Y-%m-%d'),
                'Target': future_targets
            })
            st.subheader("Next 5-Day Target Prices")
            st.table(target_table)

            # Recursive predictions plot
            st.subheader("Recursive Predictions for Next 5 Days")
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=dates_train, y=train_predictions, mode='lines', name='Train Prediction', line=dict(color="orange")))
            fig2.add_trace(go.Scatter(x=dates_val, y=val_predictions, mode='lines', name='Validation Prediction', line=dict(color="red")))
            fig2.add_trace(go.Scatter(x=dates_test, y=test_predictions, mode='lines', name='Test Prediction', line=dict(color="pink")))
            fig2.add_trace(go.Scatter(x=future_dates, y=future_targets, mode='lines', name='Recursive Prediction', line=dict(color="gold")))

            fig2.update_layout(title="Recursive Predictions", xaxis_title="Date", yaxis_title="Price")
            st.plotly_chart(fig2)

    else:
        st.warning("Please enter a stock symbol.")
