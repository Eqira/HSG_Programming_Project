This program predicts the development of a company’s share price (based on closing prices) over the next x years using Long Short-Term Memory (LSTM) with Keras. Specifically, the code follows the subsequent steps: 

This program predicts a company's today's closing price using Long Short-Term Memory (LSTM) with Keras. Specifically, the code follows the subsequent steps: 

  (1) First, the user is asked to input the corresponding ticker symbol of the stock to be analyzed 

  (2) Using Yahoo Finance, the code then retrieves five years of historical data for the selected stock which is used to train and test an LSTM model. (Welche Zeiträume werden in train und welche in test sortiert?) 

  (3) Based on the trained model, predictions for the last x months are made, which are then plotted against the true values (welcher Zeitraum wird dargestellt?). Besides, the Root Mean Squared Error (RMSE) and the Mean Absolute Percentage Error (MAPE) are calculated and displayed to allow the user the evaluation of the model's performance. Lastly, the predicted closing price of the current day is returned, in addition to the opening as well as the high and low price of the stock.

(3) Lastly, the predictions are plotted against the true values (welcher Zeitraum wird dargestellt?). On top of this, the Root Mean Squared Error (RMSE) and the Mean Absolute Percentage Error (MAPE) are calculated and displayed to allow the user the evaluation of the model's performance. Based on the trained data, the predicted closing price of the current day is returned, in addition to the opening as well as the high and low price of the stock.

#schöner machen, output Sätze anders formulieren , auf 2 Nachkommastellen begrenzen, bei Output muss noch hin für welches Datum der closing price gilt

**User Guide:**
  (1) Run the code
  (2) Input a valid ticker symbol for the stock you aim to analyze when prompted
  (3) Wait for the model to be trained and predictions to be made
  (4) View the prediction plot and evaluation metrics as well as the estimated closing price of your selected stock 

Dependencies: matplotlib yahoo_fin yfinance pandas numpy seaborn Keras scikit-learn

Tips: Make sure you have all the dependencies installed. Double-check that you have entered a valid ticker symbol.


### old

Introduction:
This code predicts the future stock prices (closing price) of a company using Long Short-Term Memory (LSTM) with Keras. The user is asked to input a ticker symbol for the stock they want to analyze. The code then retrieves 5 years of historical data for the stock using Yahoo Finance's API and processes the data for use in training and testing an LSTM model. The model is then trained on the data and used to make predictions on the test set. The predictions are plotted against the true values and the Root Mean Squared Error (RMSE) and Mean Absolute Percentage Error (MAPE) are calculated to evaluate the model's performance. The code then returns the current Open, High and Low of the selected stock, as well as the predicted closing price of the day.

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
