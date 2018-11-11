import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# create imput obkect which reads data from MNIST datasets. Perform one_hot encoding to define the digit
mnist = input data.read_data.read_data_sets("MNIST data/", one_hot=true)

# Using interactive session makes it the default sessions so we do not need to pass sess
sess = tf.InteractiveSession()

# Define placeholders for MNIST input data
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
