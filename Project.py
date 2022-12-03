### This is the main file for our project ##
# if you have not done so yet you will need to install quandl in the Anaconda prompt with the command "anaconda install quandl"

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt
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
from yahoo_fin import stock_info as si

# gather stock symbols from major US exchanges https://levelup.gitconnected.com/how-to-get-all-stock-symbols-a73925c16a1b
df1 = pd.DataFrame( si.tickers_sp500() )
df2 = pd.DataFrame( si.tickers_nasdaq() )
df3 = pd.DataFrame( si.tickers_dow() )
df4 = pd.DataFrame( si.tickers_other() )

# convert DataFrame to list, then to sets
sym1 = set( symbol for symbol in df1[0].values.tolist() )
sym2 = set( symbol for symbol in df2[0].values.tolist() )
sym3 = set( symbol for symbol in df3[0].values.tolist() )
sym4 = set( symbol for symbol in df4[0].values.tolist() )

# join the 4 sets into one. Because it's a set, there will be no duplicate symbols
symbols = set.union( sym1, sym2, sym3, sym4 )

# Some stocks are 5 characters. Those stocks with the suffixes listed below are not of interest.
my_list = ['W', 'R', 'P', 'Q']
del_set = set()
sav_set = set()

for symbol in symbols:
    if len( symbol ) > 4 and symbol[-1] in my_list:
        del_set.add( symbol )
    else:
        sav_set.add( symbol )

print( f'Removed {len( del_set )} unqualified stock symbols...' )
print( f'There are {len( sav_set )} qualified stock symbols...' )

#Ask user for stock input

while True:
    ticker_name = input('Which ticker do you want to predict?: ')
    if ticker_name != '': #If there was a valid input, continue.
        ticker_name = ticker_name.upper().strip()
        if ticker_name in sav_set:
             ticker_input = ticker_name
             break
        else: 
            print('Please enter a valid Ticker')
            continue
    else: #If the input was an empty string, ask for the name again.
        print('Please enter a valid Ticker')
        continue

#Ticker as input -> Later we should add an input for the user where he can input which ticker they want themselve, also check if that is a valid ticker
#ticker_input = 'AAPL'

# Get the data for the stock AAPL
ticker = yf.Ticker(ticker_input)
df = ticker.history(period="5y")

#reoder the df (so that its 2022 -> 2017)
df = df.iloc[::-1]

#rename close to price
df = df.rename(columns={'Close': 'Price'})

#defince target and feature
target_y = df["Price"]
X_feat = df.iloc[:,0:3]

#Feature Scaling
sc = StandardScaler()
X_ft = sc.fit_transform(X_feat.values)
X_ft = pd.DataFrame(columns = X_feat.columns, data = X_ft, index = X_feat.index)

timesplit = TimeSeriesSplit(n_splits = 10)
for train_index, test_index in timesplit.split(X_feat):
    X_train, X_test = X_feat[:len(train_index)], X_feat[len(train_index): (len(train_index)+len(test_index))]
    y_train, y_test = target_y[:len(train_index)].values.ravel(), target_y[len(train_index): (len(train_index)+len(test_index))].values.ravel()

#Process the data for LSTM
trainX =np.array(X_train)
testX =np.array(X_test)
X_train = trainX.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test = testX.reshape(X_test.shape[0], 1, X_test.shape[1])

#Building the LSTM Model
lstm = Sequential()
lstm.add(LSTM(32, input_shape=(1, trainX.shape[1]), activation="relu", return_sequences=False))
lstm.add(Dense(1))
lstm.compile(loss="mean_squared_error", optimizer="adam")
lstm.summary()

#fit the model
history = lstm.fit(X_train, y_train,
                    epochs = 100, batch_size = 4,
                    verbose = 2, shuffle = False)

#performance eval
y_pred = lstm.predict(X_test)

print(X_test)
print(y_test)
#compare
#Predicted vs True Adj Close Value – LSTM
plt.plot(y_test, label="True Value")
plt.plot(y_pred, label="LSTM Value")
plt.title("Prediction by LSTM")
plt.xlabel("Time Scale")
plt.ylabel("Scaled USD")
plt.legend()
plt.show()