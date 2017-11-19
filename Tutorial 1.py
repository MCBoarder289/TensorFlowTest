# https://www.tensorflow.org/get_started/get_started

import tensorflow as tf


# Creating constant notes, and then adding them together
node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0)  # also tf.float32 implicitly
print(node1, node2)

sess = tf.Session()
print(sess.run([node1, node2]))


# from __future__ import print_function  # Don't need this function, Python 2.X used to allow "print "word""
# the above from __future__ statement is forcing print to be a function (see link below)
# https://stackoverflow.com/questions/32032697/how-to-use-from-future-import-print-function


node3 = tf.add(node1, node2)
print("node3:", node3)
print("sess.run(node3):", sess.run(node3))

# Using Placeholders for values instead of constants, and then making an addition node/function

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b  # + provides a shortcut for tf.add(a, b)

print(sess.run(adder_node, {a: 3, b: 4.5}))
print(sess.run(adder_node, {a: [1, 3], b: [2, 4]}))

add_and_triple = adder_node * 3.
print(sess.run(add_and_triple, {a: 3, b: 4.5}))

# Building a linear model (think y = Mx + b), and providing initial variables (0.3 and -0.3)


W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W*x + b

# Then we initialize those variables, and run that session
init = tf.global_variables_initializer()
sess.run(init)

# Shows results when x is any of the fed values, and returns a vector of what y is based on x inputs
print(sess.run(linear_model, {x: [1, 2, 3, 4]}))

# Creating a placeholder "y" to produce what our EXPECTED values are (what we're trying to achieve)
# Then we need to create a loss function (for linear regression, a standard loss model sums the squares of the deltas
# between the current model and the provided data.

# Loss function is doing math of creating a vector subtracting y from the linear model to get the example error's delta
# and subsequently squaring those errors, and then summing the squared errors to produce a single scalar
# that abstracts the error from all examples (0 would be perfect).

y = tf.placeholder(tf.float32)  # expected values
squared_deltas = tf.square(linear_model - y)  # squared deltas (part of loss function)
loss = tf.reduce_sum(squared_deltas)  # full loss function = summing squared deltas
print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))  # Loss value is 23.66

# Manually improving the model to make W = -1 and b = 1 to have linear_model return zero error
# tf.assign reassigns the variable in the model

fixW = tf.assign(W, [-1.])
fixb = tf.assign(b, [1.])
sess.run([fixW, fixb])
print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))

# Using Machine Learning to optimize the model with expected values. TensorFlow's optimizers slowly change each variable
# in order to minimize the loss function. The simplest optimizer is gradient descent, which modifies
# each variable according to the magnitude of the derivative of loss with respect to that variable

optimizer = tf.train.GradientDescentOptimizer(0.01)  # Choosing optimizer type
train = optimizer.minimize(loss)  # Giving it a goal (minimize the loss function)

sess.run(init)  # reset values to incorrect defaults
for i in range(1000):
    sess.run(train, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]})

print(sess.run([W, b]))





