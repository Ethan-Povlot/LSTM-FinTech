#imports
import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

df = np.asarray(pd.read_csv("final_dataset(date).csv", parse_dates=True))
output = np.asarray(pd.read_csv("output.csv"))

epochs = 7
batch_size = 1

#scale the data
scaler = MinMaxScaler(feature_range=(0,1))

#split data into x and y train datasets
x_train = df[:int(df.shape[0]*0.8)]
x_test = df[int(df.shape[0]*0.8):]
y_train = output[:int(output.shape[0]*0.8)]
y_test = output[int(output.shape[0]*0.8):]

#convert x and y train ti numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)

#reshape data
x_train = np.reshape(x_train, (1, x_train.shape[0], x_train.shape[1]))
y_train = np.reshape(y_train, (1, y_train.shape[0], y_train.shape[1]))


print(x_train.shape)
#build LSTM model
model = Sequential()
model.add(LSTM(4500, return_sequences=True, input_shape=(4023, 11)))
model.add(Dropout(0.2))
model.add(LSTM(2500, return_sequences=True))
model.add(Dropout(0.1))
model.add(LSTM(1500, return_sequences=True))
model.add(Dropout(0.1))
model.add(LSTM(1000, return_sequences=False))
#first argument is numberof neurons
model.add(Dense(1))

#compile model
model.compile(optimizer='adam', loss='mean_squared_error')

#train the model
model.fit(x_train, y_train, batch_size=32, epochs=epochs)
model.summary()
model.save_weights('weights.h5')


#reshape data
x_test = np.reshape(x_test, (1, x_test.shape[0], x_test.shape[1]))
y_test = np.reshape(y_test, (1, y_test.shape[0], y_test.shape[1]))

#get predicted price values
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

#visualize data
plt.figure(figsize=(16,8))
plt.title('Actual vs Predicted Close Price History')
plt.plot(output, color='red', marker='Actual')
plt.plot(predictions, color='blue', marker='Predicted')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Price', fontsize=18)
plt.show()

#evaluate model so find root mean squared error
#0 is perfect
rmse = 0.5 * (np.mean(predictions - output)**2)
rmse