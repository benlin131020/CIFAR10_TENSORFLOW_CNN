from time import time
import numpy as np
import tensorflow as tf
from data.cifar10 import load_cifar10

INPUT_SHAPE = 32 * 32 * 3
NUM_CLASSES = 10
NUM_TRAIN_DATA = 50000
NUM_TEST_DATA = 10000
BATCH_SIZE = 100
NUM_BATCHES = NUM_TRAIN_DATA // BATCH_SIZE
EPOCHS = 100
LEARNING_RATE = 0.0001

def weight(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1), name='W')

def bias(shape):
    return tf.Variable(tf.constant(0.1, shape=shape), name='b')

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = load_cifar10()
    with tf.name_scope("InputLayer"):
        x = tf.placeholder(tf.float32, [None, 32, 32, 3], name='x')
        y = tf.placeholder(tf.float32, shape=[None, NUM_CLASSES], name='y')
        batch_size = tf.placeholder(tf.int64)
        #Using tf.data.Dataset for batching.
        #train_batch_dataset for training.
        train_batch_dataset = tf.data.Dataset.from_tensor_slices((x, y)).batch(batch_size).repeat()
        #train_dataset and test_dataset for calculating accuracy and loss
        #Always batch even if you want to one shot it.
        train_dataset = tf.data.Dataset.from_tensor_slices((x, y)).batch(batch_size)
        test_dataset = tf.data.Dataset.from_tensor_slices((x, y)).batch(batch_size)
        #Initializing iterator to get the data inside the dataset.
        iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
        train_batch_init_op = iterator.make_initializer(train_batch_dataset)
        train_init_op = iterator.make_initializer(train_dataset)
        test_init_op = iterator.make_initializer(test_dataset)
        features, y_label = iterator.get_next()
    with tf.name_scope("Conv1"):
        w1 = weight([5, 5, 3, 16])
        b1 = bias([16])
        conv1 = tf.nn.relu(conv2d(features, w1) + b1)
    with tf.name_scope("MaxPool1"):
        pool1 = max_pool_2x2(conv1)
    with tf.name_scope("Conv2"):
        w2 = weight([5, 5, 16, 32])
        b2 = bias([32])
        conv2 = tf.nn.relu(conv2d(pool1, w2) + b2)
    with tf.name_scope("MaxPool2"):
        pool2 = max_pool_2x2(conv2)
    with tf.name_scope("Flatten"):
        flat1 = tf.layers.Flatten()(pool2)
    with tf.name_scope("FC1"):
        w3 = weight([int(flat1.shape[1]), 128])
        b3 = bias([128])
        hidden1 = tf.nn.relu(tf.matmul(flat1, w3) + b3)
        dropout1 = tf.nn.dropout(hidden1, rate=0.2)
    with tf.name_scope("FC2"):
        w4 = weight([128, 256])
        b4 = bias([256])
        hidden2 = tf.nn.relu(tf.matmul(dropout1, w4) + b4)
        dropout2 = tf.nn.dropout(hidden2, rate=0.2)
    with tf.name_scope("OutputLayer"):
        w4 = weight([256, 10])
        b4 = bias([10])
        logits = tf.matmul(dropout2, w4) + b4
        y_predict = tf.nn.softmax(logits)
    with tf.name_scope("Loss"):
        #WARNING: This op expects unscaled logits, since it performs a softmax on logits internally for efficiency.
        #Do not call this op with the output of softmax, as it will produce incorrect results.
        cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y_label))
        tf.summary.scalar("cross entropy", cross_entropy_loss)
    with tf.name_scope("Optimizer"):
        optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cross_entropy_loss)
    with tf.name_scope("Accuracy"):
        correct_prediction = tf.equal(tf.argmax(y_predict, 1), tf.argmax(y_label, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar("accuracy", accuracy)
    #training
    startTime = time()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train_writer = tf.summary.FileWriter("logs/train", sess.graph)
        test_writer = tf.summary.FileWriter("logs/test")
        merged = tf.summary.merge_all()
        for epoch in range(EPOCHS):
            sess.run(train_batch_init_op, feed_dict={x: x_train, y: y_train, batch_size: BATCH_SIZE})
            for i in range(NUM_BATCHES):
                sess.run(optimizer)
            #training data's loss and acc
            sess.run(train_init_op, feed_dict={x: x_train, y: y_train, batch_size: NUM_TRAIN_DATA})
            loss, acc, train_result = sess.run([cross_entropy_loss, accuracy, merged])
            train_writer.add_summary(train_result, epoch)
            #testing data's loss and acc
            sess.run(test_init_op, feed_dict={x: x_test, y: y_test, batch_size: NUM_TEST_DATA})
            test_loss, test_acc, test_result = sess.run([cross_entropy_loss, accuracy, merged])
            test_writer.add_summary(test_result, epoch)
            print("Epoch%02d:" % (epoch+1), "loss:{:.9f}".format(loss), "accuracy:", acc, "test_loss:{:.9f}".format(test_loss), "test_accuracy:", test_acc)

        duration = time() - startTime
        print("Training Finished takes:", duration, "secs")
