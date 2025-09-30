import os
from io import BytesIO

import streamlit as st
import pandas as pd
from neuralprophet import NeuralProphet
import matplotlib.pyplot as plt
import torch

# Set the random seed for reproducibility
torch.manual_seed(42)


def preprocess_data(data, ticker):
    # Convert 'Date' to datetime
    data['Date'] = pd.to_datetime(data['Date'], utc=True)
    #st.write(f"Data shape before proceeding: {data.shape}")

    # Filter data for the selected ticker
    ticker_data = data[data['Ticker'] == ticker]

    # Remove duplicates by keeping the row with the highest 'Volume' for each 'Date'
    ticker_data = ticker_data.sort_values(['Date', 'Volume'], ascending=[True, False]).drop_duplicates(subset=['Date'],
                                                                                                       keep='first')
    #st.write(f"Ticker data shape after removing duplicates: {ticker_data.shape}")

    # Rename 'Date' to 'ds' and 'Close' to 'y' for NeuralProphet
    ticker_data = ticker_data[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})

    if ticker == "ADDYY":
        ticker_data = ticker_data[ticker_data['ds'] >= '2015-01-01']
    return ticker_data


def forecast(data, epochs):
    # begin
    n = len(data)
    # Set the test period to the last 90 days
    test_size = 90
    # Define the test set as the last 90 days
    test_data = data.iloc[-test_size:]

    # Split the remaining data (excluding the test set) into training and validation
    train_valid_data = data.iloc[:n - test_size]

    # Determine the split for training and validation (80% training, 20% validation)
    train_size = 0.80
    train_end = int(train_size * len(train_valid_data))

    train_data = train_valid_data.iloc[:train_end]
    valid_data = train_valid_data.iloc[train_end:]
    # end
    model: NeuralProphet = NeuralProphet(trend_reg=0.0001, yearly_seasonality=True, weekly_seasonality=True,
                                         daily_seasonality=False,
                                         learning_rate=0.001, seasonality_mode='multiplicative')

    # Train the model on training data and validate on validation data
    # freq='B' tells NeuralProphet to expect data only on business days (excluding weekends)
    metrics = model.fit(train_data, validation_df=valid_data, freq='B', epochs=epochs, early_stopping=True)

    # After training, test the model on the test data
    forecast_test = model.predict(test_data)  # Predict on the test set
    return forecast_test, model, metrics


def plot_training_loss(metrics):
    # Extract training and validation loss from metrics
    training_loss = metrics['Loss'].values
    validation_loss = metrics['Loss_val'].values

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(training_loss, label="Training Loss")
    ax.plot(validation_loss, label="Validation Loss")
    ax.set_title("Training vs Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    st.pyplot(fig)


# Streamlit App
st.title("Stock Prediction with NeuralProphet")
st.info("Test stock data is auto loaded. Please select a ticker to start forecasting.")


def get_default_file():
    try:
        with open('World-Stock-Prices-Dataset.csv', 'rb') as f:
            return BytesIO(f.read())
    except FileNotFoundError:
        return None


uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

# If no file is uploaded, use the default file
if uploaded_file is None:
    uploaded_file = get_default_file()
    if uploaded_file is not None:
        st.info("Using default dataset: World-Stock-Prices-Dataset.csv")
    else:
        st.warning("Default dataset World-Stock-Prices-Dataset.csv not found.")

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    tickers = data['Ticker'].unique()
    #get the index of Adidas ticker ADDYY
    default_index = tickers.tolist().index("ADDYY")
    ticker = st.selectbox("Select Ticker", tickers, index=default_index)

    data_processed = preprocess_data(data, ticker)

    epochs = st.slider("Select Epochs", 10, 200, 200, step=10)

    if st.button("Train Model and Forecast"):
        forecast_data, model, metrics = forecast(data_processed, epochs)
        fig = model.plot(forecast_data)
        st.plotly_chart(fig)
        plot_training_loss(metrics)
        fig_components = model.plot_components(forecast_data)
        st.plotly_chart(fig_components)
