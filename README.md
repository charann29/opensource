
## "Trading using Technical and Timeseries Analysis"


# ML Model Datasets Using Streamlits 
This repository contains my machine learning models implementation code using streamlit in the Python programming language.<br><br>
<a href = "https://ml-model-datasets-using-apps-3gy37ndiancjo2nmu36sls.streamlit.app/"><img width="960" title = "Website Image" alt="Website Image" src="https://github.com/madhurimarawat/ML-Model-Datasets-Using-Streamlits/assets/105432776/64584f95-f19c-426d-b647-a6310d0f0d2d"></a>
<br><br>
<a href = "https://ml-model-datasets-using-apps-3gy37ndiancjo2nmu36sls.streamlit.app/"><img width="960" title = "Website Image" alt="Website Image" src="https://github.com/madhurimarawat/ML-Model-Datasets-Using-Streamlits/assets/105432776/3c872711-3e6c-4d4a-a216-4cf5b7d98361"></a>


## Streamlit application deploy link : https://strmlit-jat-automated-technical-analysis.streamlit.app/

## Project goal:
Profitable stocks and crypto trading involves a lot of know how and experience in Technical Analysis. However, the fundamentals behind technical analysis techniques, tools, resources and effective strategies can be complex to grasp, understand and even expensive to access.

## Solution:
The main point of any sort of asset trading is to make a profit. This boils down to effectively three actions based on the price movements, **&#39;When should I buy?&#39;**, **&#39;When should I sell?&#39;** and **&#39;When should I hold my current position?&#39;** to maximize profits and minimize losses. Therefore, by using data analytics it was possible to translate real-time price movements to determine whether to buy, sell or hold based on historical price trends. This was achieved by combining a number of popularly used trading strategies and indicators such as **&#39;Moving Average Convergence Divergence&#39;**, **&#39;Slow Stochastic&#39;**, **&#39;Relative Strength Index&#39;** etc. More so, by feeding these sequences to a **Transformer Encoder Neural Network** to learn the price patterns and trading actions, the deep learning model could provide with the most appropriate action to be taken at any given time.

## Libraries : 
These are the libraries used to build the project - 
h5py==3.6.0   ,  html5lib==1.1  ,  Keras   ,   numpy  ,  pandas  ,  plotly  ,  requests==2.31.0  ,  scikit-learn==1.0.1  ,  streamlit  ,  tensorflow-cpu  ,  yfinance

## Project Overview

This project is an advanced trading analysis tool that combines technical indicators with machine learning models to predict price movements and suggest trading actions across various financial markets. It supports multiple asset classes including stocks, forex, futures, cryptocurrencies, and market indexes.

## Key Features

### Multi-Asset Support
- Analyze stocks from major indices (S&P 500, NASDAQ 100, Dow Jones, FTSE 100, DAX, CAC 40, etc.)
- Forex pairs
- Futures and commodities
- Cryptocurrencies (via Binance)
- Market indexes


### Comprehensive Technical Analysis
- Moving Average Convergence Divergence (MACD)
- Relative Strength Index (RSI)
- Stochastic Oscillator
- Simple and Exponential Moving Averages
- On-Balance Volume (OBV)
- Average True Range (ATR)
- Pivot Points


### Advanced Prediction Models
- Price prediction using time series forecasting
- Trading action recommendations (Buy, Hold, Sell)
- Models: action_prediction_model.h5 and price_prediction_model.h5

### Flexible Time Intervals
Analyze data in various timeframes:
- 1 minute to 1 week for cryptocurrencies
- 5 minutes to 1 week for other assets

### Interactive Visualizations
- Historical price action charts
- Predicted price movements
- Technical indicator graphs

### Risk Assessment
Adjustable trading volatility preferences:
- Low
- Medium
- High

### Automated Data Updates
Regular updates of market data to ensure accuracy in analysis

## How It Works

1. **Data Sourcing**: 
   - Fetches real-time and historical data from Binance (for cryptocurrencies) and Yahoo Finance (for other assets)
   - Uses data_sourcing.py and update_market_data.py to keep information current

2. **Technical Analysis**: 
   - Calculates various technical indicators using technical_indicators.py
   - Implements complex calculations for advanced market insights

3. **Machine Learning Predictions**: 
   - Utilizes trained models to forecast price movements and suggest trading actions
   - Implements both price and action prediction models

4. **Visualization**: 
   - Creates interactive charts using graph.py
   - Displays historical data, predictions, and technical indicators

5. **User Interface**: 
   - Streamlit-based interface (Trade.py) for easy interaction
   - Allows users to select assets, time intervals, and risk preferences

6. **Analysis and Recommendations**: 
   - Provides current price, recent price changes
   - Offers trading action recommendations with confidence levels
   - Estimates forecast prices for selected time intervals

7. **Risk Management**: 
   - Suggests buy and sell prices based on user's risk preference
   - Utilizes pivot points for different risk levels

## Project Structure


- `Trade.py`: Main application file with Streamlit interface
- `data_sourcing.py`: Handles data retrieval from various sources
- `technical_indicators.py`: Calculates technical indicators
- `model.py`: Implements machine learning models for predictions
- `graph.py`: Generates interactive visualizations
- `scaling.py`: Preprocesses and scales data for analysis
- `update_market_data.py`: Keeps market data up-to-date
- `indicator_analysis.py`: Analyzes indicator signals


## OUTPUTS

<img width="1440" alt="image" src="https://github.com/Jatavedreddy/cmr_opensource_j/assets/165547397/5dcce83a-a724-40e1-a952-7cd736221cf1">

<img width="1440" alt="image" src="https://github.com/Jatavedreddy/cmr_opensource_j/assets/165547397/bfa30caa-1677-4b4d-a89e-d5d68c447974">

<img width="1440" alt="image" src="https://github.com/Jatavedreddy/cmr_opensource_j/assets/165547397/c173ca6b-511c-4e45-8a13-9320b49b192a">

<img width="1440" alt="image" src="https://github.com/Jatavedreddy/cmr_opensource_j/assets/165547397/8a5af0bd-9813-4774-b46b-33a2540a726d">

<img width="1440" alt="image" src="https://github.com/Jatavedreddy/cmr_opensource_j/assets/165547397/57419e7f-d7eb-4217-9560-7c65baa84756">

<img width="1440" alt="image" src="https://github.com/Jatavedreddy/cmr_opensource_j/assets/165547397/b1d4b4d0-22f8-4978-8211-60fbcf11bbaa">

<img width="1440" alt="image" src="https://github.com/Jatavedreddy/cmr_opensource_j/assets/165547397/9a4293d1-0183-4613-89ab-61e1852051de">

<img width="1440" alt="image" src="https://github.com/Jatavedreddy/cmr_opensource_j/assets/165547397/1fd3ed51-f8e9-4528-a041-830b63ef07d0">

## Disclaimer

This tool is for informational purposes only. It is not intended to be investment advice. Always do your own research and consult with a licensed financial advisor before making any investment decisions.
Footer
