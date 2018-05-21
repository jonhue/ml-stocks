import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


# Import data
data = pd.read_csv('data.csv')
## Drop date variable
data = data.drop(['DATE'], 1)
## Dimensions of datasheet
n = data.shape[0]
p = data.shape[1]
## Make data a np.array
data = data.values
## Training & test data
train_start = 0
train_end = int(np.floor(0.8*n))
test_start = train_end + 1
test_end = n
data_train = data[np.arange(train_start, train_end), :]
data_test = data[np.arange(test_start, test_end), :]
## Scale data
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler.fit(data_train)
data_train = scaler.transform(data_train)
data_test = scaler.transform(data_test)
## Build X & Y
x_train = data_train[:, 1:]
y_train = data_train[:, 0]
x_test = data_test[:, 1:]
y_test = data_test[:, 0]


# Initializers
sigma = 1
weight_initializer = tf.variance_scaling_initializer(mode="fan_avg", distribution="uniform", scale=sigma)
bias_initializer = tf.zeros_initializer()


# Model architecture
n_inputs = 500
n_hidden_1 = 1024
n_hidden_2 = 512
n_hidden_3 = 256
n_hidden_4 = 128
n_output = 1


# Placeholders
X = tf.placeholder(dtype=tf.float32, shape=[None, n_inputs])
Y = tf.placeholder(dtype=tf.float32, shape=[None])


# Network topology

## Layer 1
weight_hidden_1 = tf.Variable(weight_initializer([n_inputs, n_hidden_1]))
bias_hidden_1 = tf.Variable(bias_initializer([n_hidden_1]))
## Layer 2
weight_hidden_2 = tf.Variable(weight_initializer([n_hidden_1, n_hidden_2]))
bias_hidden_2 = tf.Variable(bias_initializer([n_hidden_2]))
## Layer 3
weight_hidden_3 = tf.Variable(weight_initializer([n_hidden_2, n_hidden_3]))
bias_hidden_3 = tf.Variable(bias_initializer([n_hidden_3]))
## Layer 4
weight_hidden_4 = tf.Variable(weight_initializer([n_hidden_3, n_hidden_4]))
bias_hidden_4 = tf.Variable(bias_initializer([n_hidden_4]))
## Output
weight_output = tf.Variable(weight_initializer([n_hidden_4, n_output]))
bias_output = tf.Variable(bias_initializer([n_output]))


# Model calculations

## Layer 1
hidden_1 = tf.nn.relu(tf.add(tf.matmul(X, weight_hidden_1), bias_hidden_1))
## Layer 2
hidden_2 = tf.nn.relu(tf.add(tf.matmul(hidden_1, weight_hidden_2), bias_hidden_2))
## Layer 3
hidden_3 = tf.nn.relu(tf.add(tf.matmul(hidden_2, weight_hidden_3), bias_hidden_3))
## Layer 4
hidden_4 = tf.nn.relu(tf.add(tf.matmul(hidden_3, weight_hidden_4), bias_hidden_4))
## Output
output = tf.nn.relu(tf.add(tf.matmul(hidden_4, weight_output), bias_output))


# Cost function
mse = tf.reduce_mean(tf.squared_difference(output, Y))


# Optimizer
optimizer = tf.train.AdamOptimizer().minimize(mse)


# Training

## Initialize session
graph = tf.Session()
graph.run(tf.global_variables_initializer())

# ## Setup interactive plot
# plt.ion()
# fig = plt.figure()
# ax1 = fig.add_subplot(111)
# line1, = ax1.plot(y_test)
# line2, = ax1.plot(y_test*0.5)
# plt.show()

## Epochs & batch size
epochs = 10
batch_size = 256

for e in range(epochs):

    ## Shuffle training data
    shuffle_indices = np.random.permutation(np.arange(len(y_train)))
    x_train = x_train[shuffle_indices]
    y_train = y_train[shuffle_indices]

    ## Minibatch training
    for i in range(0, len(y_train) // batch_size):
        start = i * batch_size
        batch_x = x_train[start:start + batch_size]
        batch_y = y_train[start:start + batch_size]
        ## Run optimizer with batch
        graph.run(optimizer, feed_dict={X: batch_x, Y: batch_y})

        # ## Show progress
        # if np.mod(i, 5) == 0:
        #     ## Prediction
        #     prediction = graph.run(output, feed_dict={X: x_test})
        #     line2.set_ydata(prediction)
        #     plt.title('Epoch ' + str(e) + ', Batch ' + str(i))
        #     file_name = 'img/epoch_' + str(e) + '_batch_' + str(i) + '.png'
        #     plt.savefig(file_name)
        #     plt.pause(0.01)

# Print final MSE after training
mse_final = graph.run(mse, feed_dict={X: x_test, Y: y_test})
print(mse_final)
