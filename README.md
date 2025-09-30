---
title: Stock Prediction Neural Prophet
emoji: 📉
colorFrom: purple
colorTo: green
sdk: streamlit
sdk_version: 1.42.2
app_file: app.py
pinned: false
license: apache-2.0
short_description: Stock prediction with Neural Prophet
---

# Stock Prediction with NeuralProphet

A Streamlit web application that uses NeuralProphet to forecast stock prices based on historical data.

## Features

- **Interactive Stock Selection**: Choose from multiple stock tickers in the dataset
- **Time Series Forecasting**: Uses NeuralProphet with yearly and weekly seasonality for accurate predictions
- **Train/Validation/Test Split**: Automatically splits data into 80% training, 20% validation, and 90-day test period
- **Visualization**:
  - Forecast plots showing predicted vs actual values
  - Training vs validation loss curves
  - Component plots (trend, seasonality)
- **Configurable Training**: Adjustable epoch count (10-200) via slider
- **Early Stopping**: Prevents overfitting during training

## Model Configuration

The NeuralProphet model is configured with:
- Trend regularization: 0.0001
- Yearly and weekly seasonality enabled
- Multiplicative seasonality mode
- Learning rate: 0.001
- Business day frequency (excludes weekends)

## Usage

1. Upload your own CSV file or use the default `World-Stock-Prices-Dataset.csv`
2. Select a stock ticker from the dropdown
3. Adjust the number of training epochs (default: 200)
4. Click "Train Model and Forecast" to generate predictions

## Data Requirements

CSV file should contain:
- `Date`: Stock date
- `Ticker`: Stock symbol
- `Close`: Closing price
- `Volume`: Trading volume (used for duplicate resolution)

## Technical Details

- Built with Streamlit for the web interface
- Uses PyTorch for model training (seed: 42 for reproducibility)
- Handles duplicate dates by keeping the entry with highest volume
- Special preprocessing for ADDYY ticker (filters data from 2015-01-01 onwards)
