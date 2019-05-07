import numpy as np
import tensorflow as tf
from tensorflow.python.keras.datasets.cifar10 import load_data
from sklearn.model_selection import train_test_split

NUM_CLASSES = 10

def load_cifar10(normalize=True, one_hot=True, label_reshape=True, validation=False, validation_state=0.2):
    r'''
    Loading cifar10 by `tf.keras`.

    If `label_reshape` is `True`,
    the shape of `y_train` and `y_test` will be `(-1,10)`,
    else `(-1,1,10)`

    Returns:
      (x_train, y_train), (x_test, y_test)
    '''
    (x_train, y_train), (x_test, y_test) = load_data()
    if normalize:
        x_train = x_train.astype(np.float32)
        x_train /= 255.0
        x_test = x_test.astype(np.float32)
        x_test /= 255.0
    if one_hot:
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            y_train = sess.run(tf.one_hot(y_train, NUM_CLASSES))
            y_test = sess.run(tf.one_hot(y_test, NUM_CLASSES))
    if label_reshape:
        y_train = y_train.reshape(-1, NUM_CLASSES)
        y_test = y_test.reshape(-1, NUM_CLASSES)
    if validation:
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=validation_state, random_state=42)
        return (x_train, y_train), (x_test, y_test), (x_val, y_val)
    return (x_train, y_train), (x_test, y_test)