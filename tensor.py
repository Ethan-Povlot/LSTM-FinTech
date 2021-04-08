import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, losses, optimizers, Sequential, preprocessing, activations
from keras.utils import plot_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import re

scaler = MinMaxScaler(feature_range=(0, 1))
features_df = pd.read_csv("final_dataset(date).csv")

features_np = np.array(features_df)

features = scaler.fit_transform(features_np)

num_features = features_np.shape[1]
batch_size = 1

# [samples, time steps, features] 
# samples is the number of data points we have
# time steps is the number of time-dependent steps that are there in a single data point
# features is the number of variables we have for the corresponding true value in Y

labels_np = np.array(pd.read_csv("output.csv"))
labels = scaler.fit_transform(labels_np)

x_train = features[int(features_np.shape[0] * 0.75):]
x_test = features[:int(features_np.shape[0] * 0.25)]

y_train = labels[int(labels_np.shape[0] * 0.75):]
y_test = labels[:int(labels_np.shape[0] * 0.25)]

x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
y_train = np.reshape(y_train, (y_train.shape[0], 1, y_train.shape[1]))
x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))
y_test = np.reshape(y_test, (y_test.shape[0], 1, y_test.shape[1]))

model = Sequential()
model.add(layers.LSTM(units=5000, return_sequences=True, input_shape=(1, num_features)))
model.add(layers.Dropout(0.2))
model.add(layers.LSTM(units=1000, return_sequences=True))
model.add(layers.Dropout(0.1))
model.add(layers.LSTM(units=1000, return_sequences=True))
model.add(layers.Dropout(0.1))
model.add(layers.LSTM(units=1000))
model.add(layers.Dense(1))

model.summary()
ADAM=optimizers.Adam(learning_rate=0.001, beta_1=0.09, beta_2=0.999, amsgrad=False)
model.compile(loss='mean_squared_error', optimizer=ADAM)
history = model.fit(x_train, y_train,epochs=1,validation_data=(x_test, y_test),verbose=1,shuffle=False)

predictions = model.predict(x_train) #would be better as x_test - predict from the end
predictions = scaler.inverse_transform(predictions)
data = features_np
'''
with open('prediction.txt', 'r+') as file:
    for i in range(predictions.shape[0]):
        for j in range(data.shape[0]):
            file.write(f"{data[i][j]}, {predictions[i][j]}")
'''

plt.xlabel('date', fontsize=18)
plt.ylabel('close', fontsize=18)
plt.plot(data[:,4], label='Close')

plt.plot(predictions, label='Predictions')
plt.legend()
plt.show()
