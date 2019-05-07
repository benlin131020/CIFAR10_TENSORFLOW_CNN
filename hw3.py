from time import time
import numpy as np
import tensorflow as tf
from data.cifar10 import load_cifar10

NUM_INPUT = 32 * 32 * 3
NUM_CLASSES = 10
NUM_TRAIN_DATA = 50000
NUM_TEST_DATA = 10000
BATCH_SIZE = 100
NUM_BATCHES = NUM_TRAIN_DATA // BATCH_SIZE
EPOCHS = 10
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
    (x_train, y_train), (x_test, y_test), (x_val, y_val) = load_cifar10(validation=True)
    with tf.name_scope("InputLayer"):
        x = tf.placeholder(tf.float32, [None, 32, 32, 3], name='x')
        y = tf.placeholder(tf.float32, shape=[None, NUM_CLASSES], name='y')
        batch_size = tf.placeholder(tf.int64)
        dataset = tf.data.Dataset.from_tensor_slices((x, y)).batch(batch_size).repeat()
        iter = dataset.make_initializable_iterator()
        features, y_label = iter.get_next()
    with tf.name_scope("Conv1"):
        w1 = weight([5, 5, 3, 16])
        b1 = bias([16])
        conv1 = tf.nn.relu(conv2d(features, w1) + b1)
    with tf.name_scope("MaxPool1"):
        pool1 = max_pool_2x2(conv1)    
    with tf.name_scope("Conv2"):
        w2 = weight([5, 5, 16, 36])
        b2 = bias([36])
        conv2 = tf.nn.relu(conv2d(pool1, w2) + b2)
    with tf.name_scope("MaxPool2"):
        pool2 = max_pool_2x2(conv2)
    with tf.name_scope("Flatten"):
        flat1 = tf.layers.Flatten()(pool2)
    with tf.name_scope("FC1"):
        w3 = weight([int(flat1.shape[1]), 128])
        #w3 = weight([2304, 128])
        b3 = bias([128])
        hidden1 = tf.nn.relu(tf.matmul(flat1, w3) + b3)
        dropout1 = tf.nn.dropout(hidden1, rate=0.2)
    with tf.name_scope("OutputLayer"):
        w4 = weight([128, 10])
        b4 = bias([10])
        y_predict = tf.matmul(dropout1, w4) + b4
    with tf.name_scope("Optimizer"):
        #WARNING: This op expects unscaled logits, since it performs a softmax on logits internally for efficiency.
        #Do not call this op with the output of softmax, as it will produce incorrect results.
        cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_predict, labels=y_label))
        optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cross_entropy_loss)
    with tf.name_scope("Evaluation"):
        correct_prediction = tf.equal(tf.argmax(y_predict, 1), tf.argmax(y_label, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) 
    #training
    epoch_list=[]
    accuracy_list=[]
    loss_list=[]
    print("Training...")
    startTime=time()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(iter.initializer, feed_dict={x: x_train, y: y_train, batch_size: BATCH_SIZE})
        for epoch in range(EPOCHS):
            for i in range(NUM_BATCHES):
                sess.run(optimizer)
            
            loss, acc = sess.run([cross_entropy_loss, accuracy], feed_dict={features: x_val, y_label: y_val})
            print("Train Epoch:%02d" % (epoch+1), "Loss={:.9f}".format(loss), "Accuracy=", acc)

        duration =time()-startTime
        print("Training Finished takes:",duration, "secs")     
        #evalute model
        sess.run(iter.initializer, feed_dict={x: x_test, y: y_test, batch_size: NUM_TEST_DATA})
        print("Accuracy:", sess.run(accuracy))
        writer = tf.summary.FileWriter("logs/", sess.graph)
        