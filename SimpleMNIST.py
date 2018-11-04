#
# SimpleMNIST.py
#   Simple NN to classify handwritten digits from MNIST dataset
#

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# We use the TF helper function to pull down the data from MNIST site
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# x is placeholder for the 28 x 28 image data
x = tf.placeholder(tf.float32, shape=[None, 784])

# y_is called "y bar" and is a 10 element vector, containing the predicited probability of each
# digit(0-9) class. Such as [0.14, 0.8, 0,0,0,0,0,0,0,0.06]
y_ = tf.placeholder(tf.float32, [None, 10])

# define weights and balances
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# define our model
y = tf.nn.softmax(tf.matmul(x, W) + b)

# loss is cross entropy
cross_entropy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

# each training step in gradient decent we want to minimize cross_entropy
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
