# Get the data for the stock specified by the user
ticker = yf.Ticker(ticker_input)
df = ticker.history(period="5y")

# Define the target (i.e. the 'Close' column) and the features (all other columns)
# Data processing code from:  https://www.projectpro.io/article/stock-price-prediction-using-machine-learning-project/571#:~:text=The%20idea%20is%20to%20weigh,to%20predict%20future%20stock%20prices.
target_y = df["Close"]
X_feat = df.iloc[:,0:3]

# Split the data into training and testing sets using TimeSeriesSplit
# Timesplit and LSTM code from: https://www.analyticsvidhya.com/blog/2021/10/machine-learning-for-stock-market-prediction-with-step-by-step-implementation/#:~:text=Creating%20a%20Training%20Set%20and,backpropagation%20on%20the%20test%20set.

timesplit = TimeSeriesSplit(n_splits = 10)
for train_index, test_index in timesplit.split(X_feat):
    X_train, X_test = X_feat[:len(train_index)], X_feat[len(train_index): (len(train_index)+len(test_index))]
    y_train, y_test = target_y[:len(train_index)].values.ravel(), target_y[len(train_index): (len(train_index)+len(test_index))].values.ravel()

# Convert the training and testing sets to NumPy arrays and reshape them for use with LSTM
trainX = np.array(X_train)
testX = np.array(X_test)
X_train = trainX.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test = testX.reshape(X_test.shape[0], 1, X_test.shape[1])

#Building the LSTM Model
lstm = Sequential()
lstm.add(LSTM(32, input_shape=(1, trainX.shape[1]), activation="relu", return_sequences=False))
lstm.add(Dense(1))
lstm.compile(loss="mean_squared_error", optimizer="adam")
lstm.summary()

# Fit the model to the training data
history = lstm.fit(X_train, y_train,
                    epochs = 100, batch_size = 4,
                    verbose = 2, shuffle = False)

# Make predictions on the testing set 
y_pred = lstm.predict(X_test)

# Get the dates from the df dataframe
dates = df.index[len(train_index): (len(train_index)+len(test_index))]

# Plot the predicted values against the true values
plt.plot(dates ,y_test, label="True Value")
plt.plot(dates, y_pred, label="LSTM Value")
plt.title("Prediction by LSTM")
plt.xlabel("Time Scale")
plt.ylabel("Scaled USD")
plt.legend()
plt.show()

# Check the RMSE and MAPE values to evalute performance
rmse = mean_squared_error(y_test, y_pred, squared = False)
mape = mean_absolute_percentage_error(y_test, y_pred)
print("RSME: ", rmse)
print("MAPE: ", mape)

# Retrieve the data for the last day in the dataset
last_day = df.iloc[-1]

# Extract the feature values for the last day
last_day_features = last_day[0:3]

# Reshape the feature values for use with the LSTM model
last_day_features = last_day_features.values.reshape(1, 1, last_day_features.shape[0])

# Use the LSTM model to make a prediction
prediction = lstm.predict(last_day_features)

# Print today's stock information of the chosen stock
print("Today's Open, High and Low of {} is {} {} and {}".format(ticker_input, df.iloc[-1]["Open"],df.iloc[-1]["High"],df.iloc[-1]["Low"]))

# Print todays's predicted closing price of chosen stock
print("The predicted closing price is: {}".format(prediction))