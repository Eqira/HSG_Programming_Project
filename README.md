This program **predicts a company's closing price of the day** using a Long Short-Term Memory (LSTM) with Keras. Specifically, the code follows the subsequent steps: 

1. First, the user is asked to input the corresponding ticker symbol of the stock to be analyzed 

2. Using Yahoo Finance, the code then retrieves five years of historical data for the selected stock which is used to train and test an LSTM model

3. Based on the trained model, predictions for the last x months are made, which are then plotted against the true values. To allow for the user's evaluation of the model's performance, the Root Mean Squared Error (RMSE) and the Mean Absolute Percentage Error (MAPE) are calculated. Lastly, the opening, high and low price of the stock, together with the predicted closing price are printed.

**User Guide:**

 1. Run the code
  
 2. Input a valid ticker symbol for the stock you aim to analyze when prompted
  
 3. Wait for the model to be trained and predictions to be made
  
 4. View the prediction plot and evaluation metrics as well as the estimated closing price of the day of your selected stock 

**Dependencies:** matplotlib, yahoo_fin, yfinance, pandas, numpy, seaborn, Keras, scikit-learn, tabulate, datetime

**Tips:** Make sure you have all the dependencies installed. Double-check that you have entered a valid ticker symbol
