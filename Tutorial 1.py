# https://www.tensorflow.org/get_started/get_started

import tensorflow as tf

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



