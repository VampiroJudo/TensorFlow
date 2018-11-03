#
#   House_Price_Prediction.py
#
#   This is a very simple prediction of house pricess based on house size, implemented in TensorFlow.
#
#

import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation # import animation support

# Generate some house sizes between 1000 and 3500  (typical SQ Footage of house)

num_house = 160
np.random.seed(42)
house_size = np.random.randint(low=1000, high=3500, size=num_house )

# Generate some house prices for house size with a random noise added

np.random.seed(42)
house_price= house_size * 100.0 + np.random.randint(low=20000, high=70000, size=num_house )

# Plot generated house and size

plt.plot(house_size, house_price, "bx") #bx = Blue X
plt.ylabel("Price")
plt.xlabel("Size")
plt.show()

# Define number of training samples, 0.7 = 70%. We can take the first 70% since the values are randomized
num_train_samples = math.floor(num_house * 0.7)

# Define training data
train_house_size = np.array(house_size[:num_train_samples])
train_price = np.asanyarray(house_price[:num_train_samples])

train_house_size_norm = normalize(train_house_size)
train_price_norm = normalize(train_price)

#Define test data
test_house_size = np.array(house_size[num_train_samples])
test_house_price = np.array(house_price[num_train_samples])

test_house_size_norm = normalize(test_house_size)
test_house_size_norm = normalize(test_house_price)

# Set up the TensorFlow placeholders that get updated as we decend down the gradient
tf_house_size = tf.placeholder("float", name-"house_size")
tf_price = tf.placeholder("float", name="price")



tf_price_prod = tf.add(tf.multiply(tf_size_factor, tf_house_size), tf_price_offset)


#Define the loss Function (how much error) - Mean swaured error
tf_cost = tf.reduce_sum(tf.pow(tf_price_pred-tf_price, 2))/(2*num_train_samples)

#Optimzer learning rate
learning_rate = 0.1

#Define the gradient descent optimizer that will minimize the loss of defined in the operation "cost"
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(tf_cost)

# Initialize the variables
init = tf.global_variables_initializer()

# Launch the graph in the session
with tf.session() as sess
    sees.run(init)

    # Keep iterating the training data
    for interation in range(num_training_iter):

        # Fit all training data
        for(x, y) in zip(train_house_size_norm, train_price_norm):
            sess.run(optimizer, feed_dict=(tf.test_house_size: x, tf_price: y))
