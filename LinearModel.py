import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, losses, optimizers, Sequential, preprocessing, activations
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


def date_to_days(date_string):
    date_parts = i.split("/")
    return (365 * int(date_parts[2])) + (30 * int(date_parts[1])) + int(date_parts[0])


features_df = pd.read_csv("input_simple.csv")

features_np = []
for i in features_df["Date"]:
    features_np.append(date_to_days(i))

# [samples, time steps, features]
# samples is the number of data points we have
# time steps is the number of time-dependent steps that are there in a single data point
# features is the number of variables we have for the corresponding true value in Y

sc = MinMaxScaler(feature_range=(0, 1))
features = np.asarray(sc.fit_transform(np.reshape(features_np, (len(features_np), 1))))
np.random.shuffle(features)

# micah's comment: good practice to shuffle data before splitting into test/train

labels = features

print("Features shape: ", features.shape)
print("Labels shape: ", features.shape)

x_train = features[int(features.shape[0] * 0.75):]
x_test = features[:int(features.shape[0] * 0.25)]

y_train = labels[int(labels.shape[0] * 0.75):]
y_test = labels[:int(labels.shape[0] * 0.25)]

x_train = np.reshape(x_train, (x_train.shape[0], 1, 1))
y_train = np.reshape(y_train, (y_train.shape[0], 1, 1))
x_test = np.reshape(x_test, (x_test.shape[0], 1, 1))
y_test = np.reshape(y_test, (y_test.shape[0], 1, 1))

num_features = 1
batch_size = 64
epochs = 150

model = Sequential()
model.add(layers.LSTM(units=256, return_sequences=True, input_shape=(1, num_features)))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(1, activation="relu"))

# micah's comment: your previous model was huuuge, would take ages to train on my computer so I made a much smaller version to test with

model.summary()
model.compile(loss='mean_squared_error', optimizer='Adam')

# micah's comment: using default adam optimizer parameters

history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test), verbose=1,
                    shuffle=False)

predictions = model.predict(x_test)  # would be better as x_test - predict from the end

plt.plot(
    np.reshape(x_test, x_test.shape[0]),
    np.reshape(predictions, predictions.shape[0]),
    label="Predicted")

plt.plot(
    np.reshape(x_test, x_test.shape[0]),
    np.reshape(y_test, y_test.shape[0]),
    label="Truth")

plt.xlabel('Date', fontsize=14)
plt.ylabel('Close Price (F)', fontsize=14)

plt.legend()

plt.savefig("test_data.png")
plt.close()
