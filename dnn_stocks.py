from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

tf.logging.set_verbosity(tf.logging.INFO)

def dnn_model_fn(features, labels, mode):
  print(labels)
  """Model function for DNN."""
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
  # Network topology
  weight_hidden_1 = tf.Variable(weight_initializer([n_inputs, n_hidden_1]))
  bias_hidden_1 = tf.Variable(bias_initializer([n_hidden_1]))
  weight_hidden_2 = tf.Variable(weight_initializer([n_hidden_1, n_hidden_2]))
  bias_hidden_2 = tf.Variable(bias_initializer([n_hidden_2]))
  weight_hidden_3 = tf.Variable(weight_initializer([n_hidden_2, n_hidden_3]))
  bias_hidden_3 = tf.Variable(bias_initializer([n_hidden_3]))
  weight_hidden_4 = tf.Variable(weight_initializer([n_hidden_3, n_hidden_4]))
  bias_hidden_4 = tf.Variable(bias_initializer([n_hidden_4]))
  weight_output = tf.Variable(weight_initializer([n_hidden_4, n_output]))
  bias_output = tf.Variable(bias_initializer([n_output]))
  # Model calculations
  input = tf.reshape(features["x"], [-1, n_inputs])
  hidden_1 = tf.nn.relu(tf.add(tf.matmul(input, weight_hidden_1), bias_hidden_1))
  hidden_2 = tf.nn.relu(tf.add(tf.matmul(hidden_1, weight_hidden_2), bias_hidden_2))
  hidden_3 = tf.nn.relu(tf.add(tf.matmul(hidden_2, weight_hidden_3), bias_hidden_3))
  hidden_4 = tf.nn.relu(tf.add(tf.matmul(hidden_3, weight_hidden_4), bias_hidden_4))
  output = tf.nn.relu(tf.add(tf.matmul(hidden_4, weight_output), bias_output))
  # Predictions
  predictions = {
    "classes": tf.argmax(input=output, axis=1),
    "probabilities": tf.nn.softmax(output, name="softmax_tensor")
  }
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
  # Calculate loss
  loss = tf.reduce_mean(tf.squared_difference(output, labels))
  # Training optimizer
  if mode == tf.estimator.ModeKeys.TRAIN:
    # optimizer = tf.train.AdamOptimizer()
    # train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
  # Evaluation metrics
  eval_metric_ops = {
    "accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])
  }
  return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def main(unused_argv):
  # Load training and eval data
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
  data_train = np.float32(scaler.transform(data_train))
  data_test = np.float32(scaler.transform(data_test))
  ## Build X & Y
  train_data = data_train[:, 1:]
  train_labels = data_train[:, 0]
  eval_data = data_test[:, 1:]
  eval_labels = data_test[:, 0]
  # Create estimator
  mnist_classifier = tf.estimator.Estimator(model_fn=dnn_model_fn, model_dir="/tmp/stocks_deepnet_model")
  # Logging hook
  tensors_to_log = {"probabilities": "softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)
  # Train model
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": train_data},
    y=train_labels,
    batch_size=256,
    num_epochs=None,
    shuffle=True
  )
  mnist_classifier.train(input_fn=train_input_fn, steps=5000, hooks=[logging_hook])
  # Evaluate model
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": eval_data},
    y=eval_labels,
    num_epochs=10,
    shuffle=False
  )
  eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
  print(eval_results)

if __name__ == "__main__":
  tf.app.run()
