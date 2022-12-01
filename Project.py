### This is the main file for our project ##
# if you have not done so yet you will need to install quandl in the Anaconda prompt with the command "anaconda install quandl"

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import yfinance as yf
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
import datetime as dt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit, train_test_split

#Ticker as input -> Later we should add an input for the user where he can input which ticker they want themselve, also check if that is a valid ticker
ticker_list = 'AAPL'

# Get the data for the stock AAPL
ticker = yf.Ticker(ticker_list)
df = ticker.history(period="5y")

#reoder the df (so that its 2022 -> 2017)
df = df.iloc[::-1]

#rename close to price
df = df.rename(columns={'Close': 'Price'})

#add price change columns to use as features to predict the price of the next day
df['OneDayChange'] = df['Price'].shift(-1) - df['Price'].shift(-2)
df['Derivative'] = df['Price'].shift(-1) - 2*df['Price'].shift(-2) + df['Price'].shift(-3)

#remove unused columns
df = df[['Price','OneDayChange','Derivative']]

#remove NA that were created due to offset
df = df.dropna()
print(df)

target_y = df["Price"]
X_feat = df.loc[:,"OneDayChange":"Derivative"]
print(target_y)
print(X_feat)

#Feature Scaling
sc = StandardScaler()
X_ft = sc.fit_transform(X_feat.values)
X_ft = pd.DataFrame(columns = X_feat.columns, data = X_ft, index = X_feat.index)

print(X_ft)

#this is wrong -> change to time series split!!!
X_train, X_test, y_train, y_test = train_test_split(X_ft, target_y, test_size=0.2, random_state=42)

print(X_test)