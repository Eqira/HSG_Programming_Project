Introduction:
This code predicts the future stock prices (closing price) of a company using Long Short-Term Memory (LSTM) with Keras. The user is asked to input a ticker symbol for the stock they want to analyze. The code then retrieves 5 years of historical data for the stock using Yahoo Finance's API and processes the data for use in training and testing an LSTM model. The model is then trained on the data and used to make predictions on the test set. The predictions are plotted against the true values and the Root Mean Squared Error (RMSE) and Mean Absolute Percentage Error (MAPE) are calculated to evaluate the model's performance.

Dependencies:
matplotlib
yahoo_fin
yfinance
pandas
numpy
seaborn
Keras
scikit-learn

Usage:
1) Run the code.
2) Input a ticker symbol for the stock you want to analyze when prompted.
3) Wait for the model to be trained and predictions to be made.
4) The prediction plot and evaluation metrics will be displayed.

Tips:
Make sure you have all the dependencies installed.
Double-check that you have entered a valid ticker symbol.
