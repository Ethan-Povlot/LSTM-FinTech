import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

def init_data():
    input = np.asarray(pd.read_csv("input_medium.csv"))
    output = np.asarray(pd.read_csv("output.csv"))

    x_train = input[:int(input.shape[0]*0.8)]
    y_train = input[int(input.shape[0]*0.8):]
    x_test = output[:int(output.shape[0]*0.8)]
    y_test = output[int(output.shape[0]*0.8):]

    return x_train, y_train, x_test, y_test

def sigmoid(x, derivative=False):
    if (derivative == True):
        return sigmoid(x) * (1 - sigmoid(x))
    else:
        return  1 / (1 + math.exp(-x))

def tanh(x, derivative=False):
    if (derivative == True):
        return 1 - (tanh(x)^2)
    else:
        return (1 / math.cosh(x))

def LSTM(x):
    forget_gate = sigmoid(np.dot(forget_weights, x) + np.dot(recurrent_forget_weights, previous_output_state) + forget_bias)
    input_gate = sigmoid(np.dot(input_weights, x) + np.dot(recurrent_input_weights, previous_output_state) + input_bias)
    output_gate = sigmoid(np.dot(output_weights, x) + np.dot(recurrent_output_weights, previous_output_state) + output_bias)
    cell_input = tanh(np.dot(cell_weights, x) + np.dot(recurrent_cell_weights, previous_output_state) + cell_bias)

    cell_state = np.multiply(forget_gate, previous_cell_state) + np.multiply(input_gate, cell_input)
    output_state =  np.multiply(output_gate, tanh(cell_state))

    return forget_gate, input_gate, output_gate, cell_input, cell_state, output_state

def update_weights(x, h, h_previous, k):
    delta_w_forget = forget_weights + np.multiply(k, np.dot(h, (o * np.multiply(tanh(cell_state, derivative=True)^2, np.multiply(sigmoid(f, derivative=True), x) * cell_state))))
    delta_w_input =  input_weights + np.multiply(k, np.dot(h, (o * np.multiply(tanh(cell_state, derivative=True)^2, np.multiply(sigmoid(i, derivative=True), x) * c_tilde))))
    delta_w_output = output_weights + np.multiply(k, np.dot(h, np.multiply(sigmoid(o, derivative=True), x) * tanh(cell_state)))
    delta_w_cell_input = cell_weights + np.multiply(k, np.dot(h, (o * np.multiply(tanh(cell_state, derivative=True)^2, np.multiply(sigmoid(c_tilde, derivative=True), x) * i))))
    delta_u_forget =  recurrent_forget_weights + np.multiply(k, np.dot(h, (o * np.multiply(tanh(cell_state, derivative=True)^2, np.multiply(sigmoid(f, derivative=True), h_previous) * cell_state))))
    delta_u_input = recurrent_input_weights + np.multiply(k, np.dot(h, (o * np.multiply(tanh(cell_state, derivative=True)^2, np.multiply(sigmoid(i, derivative=True), h_previous) * c_tilde))))
    delta_u_output = recurrent_output_weights + np.multiply(k, np.dot(h, np.multiply(sigmoid(o, derivative=True), h_previous) * tanh(cell_state)))
    delta_u_cell_input = recurrent_cell_weights + np.multiply(k, np.dot(h, (o * np.multiply(tanh(cell_state, derivative=True)^2, np.multiply(sigmoid(c_tilde, derivative=True), h_previous) * i))))
    
    return delta_w_forget, delta_w_input, delta_w_output, delta_w_cell_input, delta_u_forget, delta_u_input, delta_u_output, delta_u_cell_input

def save():
    np.save('forget_weights.npy', forget_weights)
    np.save('input_weights.npy', input_weights)
    np.save('output_weights.npy', output_weights)
    np.save('cell_state.npy', cell_state)
    np.save('recurrent_forget_weights.npy', recurrent_forget_weights)
    np.save('recurrent_input_weights.npy', recurrent_input_weights)
    np.save('recurrent_output_weights.npy', recurrent_output_weights)
    np.save('recurrent_cell_weights.npy', recurrent_cell_weights)


input_features = 13
hidden_units = 1
learning_rate = 1

forget_weights = np.random.rand(hidden_units, input_features) #R^(h x d)
input_weights = np.random.rand(hidden_units, input_features) #R^(h x d)
output_weights = np.random.rand(hidden_units, input_features) #R^(h x d)
cell_weights = np.random.rand(hidden_units, input_features) #R^(h x d)
recurrent_forget_weights = np.random.rand(hidden_units, hidden_units) #R^(h x h)
recurrent_input_weights = np.random.rand(hidden_units, hidden_units) #R^(h x h) 
recurrent_output_weights = np.random.rand(hidden_units, hidden_units) #R^(h x h)
recurrent_cell_weights = np.random.rand(hidden_units, hidden_units) #R^(h x h)
forget_bias = 0 #R^h
input_bias = 0 #R^h
output_bias = 0 #R^h
cell_bias = 0 #R^h

x_train, y_train, x_test, y_test = init_data()
previous_cell_state = 0
previous_output_state = 0
for iteration in range(10000):
    print("Starting iteration: ", iteration, "/10000")
    f, i, o, c_tilde, cell_state, output_state = LSTM(x_train)
    forget_weights, input_weights, output_weights, cell_weights, recurrent_forget_weights, recurrent_input_weights, recurrent_output_weights, recurrent_cell_weights =  update_weights(x_train, output_state, previous_output_state, learning_rate) 
    previous_output_state = output_state

output = LSTM(x_train)[4]
error = 0.5 * (np.mean(output - y_test)**2)
print(error)
save()
