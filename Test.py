import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))

# Note that running this outputs an error message, which simply states that the cpu I have will make it slower:
# C:\tf_jenkins\home\workspace\rel-win\M\windows\PY\36\tensorflow\core\platform\cpu_feature_guard.cc:137]
# Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX AVX2
# https://stackoverflow.com/questions/47068709/your-cpu-supports-instructions-that-this-tensorflow-binary-was-not-compiled-to-u

# It will also output b'Hello, TensorFlow!', which just means it's a bytes object (vs. unicode like python 2.7)
# https://stackoverflow.com/questions/40904979/the-print-of-string-constant-is-always-attached-with-b-intensorflow

