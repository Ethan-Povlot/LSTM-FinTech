import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, losses, optimizers, Sequential, preprocessing, activations
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

num_features = 100
batch_size = 128
epochs = 150

features_np = np.asarray(pd.read_csv("final_dataset(date).csv"))
# [samples, time steps, features]
# samples is the number of data points we have / time steps is the number of time-dependent steps that are there in a single data point / features is the number of variables we have for the corresponding true value in Y

sc = MinMaxScaler(feature_range=(0, 1))
features = sc.fit_transform(features_np)

np.random.shuffle(features)

labels_np = np.asarray(pd.read_csv("output.csv"))
labels = sc.fit_transform(labels_np)

x_train = features[int(features.shape[0] * 0.75):]
x_test = features[:int(features.shape[0] * 0.25)]
y_train = labels[int(labels.shape[0] * 0.75):]
y_test = labels[:int(labels.shape[0] * 0.25)]

x_train = np.reshape(x_train, (x_train.shape[0], 1, num_features))
y_train = np.reshape(y_train, (y_train.shape[0], 1, 1))
x_test = np.reshape(x_test, (x_test.shape[0], 1, num_features))
y_test = np.reshape(y_test, (y_test.shape[0], 1, 1))

model = Sequential()
model.add(layers.LSTM(units=5000, return_sequences=True, input_shape=(1, num_features)))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(1, activation="relu"))
model.summary()
model.compile(loss='mean_squared_error', optimizer='Adam')

'''
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test), verbose=1, shuffle=False)
predictions = model.predict_on_batch(features.reshape(features.shape[0], 1, features.shape[1]))
predictions = sc.inverse_transform(predictions[:,0])
'''

history = model.fit(x_train, x_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    shuffle=True,
                    validation_data=(x_test, x_test),
                    verbose=1).history

predictions = model.predict(x_test)
print(predictions)

plt.xlabel('date', fontsize=18)
plt.ylabel('close', fontsize=18)

close = sc.inverse_transform(labels)
predictions = sc.inverse_transform(predictions)

plt.plot(close, label='Close')
plt.plot(predictions.reshape(predictions.shape[0],), label='Predictions')


plt.legend()
plt.show()
plt.close()
