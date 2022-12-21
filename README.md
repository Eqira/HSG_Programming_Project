This program allows you to **compare your prediction skills with those of a trained model** based on a Long Short-Term Memory (LSTM) with Keras. Given a stock's opening, low, and high price, you are asked to estimate its closing price on the last trading day. Your prediction and that of the model are then compared to the actual closing price. Are you able to beat the model?

The code specifically follows the subsequent steps: 

1. First, the user is asked to input the corresponding ticker symbol of the stock to be analyzed 

2. Using Yahoo Finance, the code then retrieves five years of historical data for the selected stock which is used to train and test an LSTM model

3. Based on the opening, low, and high price of the last trading day, the trained model subsequently predicts the stock's closing price, which is compared to the user's prediction and the actual closing price

4. Lastly, the model's predictions for the last five months are plotted against the true values. To allow for the user's evaluation of the model's performance, the Root Mean Squared Error (RMSE) and the Mean Absolute Percentage Error (MAPE) are calculated

**Quick User Guide:**

 1. Run the code
  
 2. Input a valid ticker symbol for the stock you aim to analyze when prompted
  
 3. Wait for the model to be trained and predictions to be made
 
 4. Enter your predicted closing price based on the stock's opening, low, and high price
  
 5. Compare your price with that of the model and find out who is better at predicting stock market prices
 
 6. View the model's prediction plot for the last five months and the corresponding evaluation metrics 

**Dependencies:** matplotlib, yahoo_fin, yfinance, pandas, numpy, seaborn, Keras, scikit-learn, tabulate, datetime

**Tips:** Make sure you have all the dependencies installed. Double-check that you have entered a valid ticker symbol
