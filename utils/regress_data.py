## Copyright 2015 Yahoo Inc.
## Licensed under the terms of the New-BSD license. Please see LICENSE file in the project root for terms.
import tensorflow as tf
from data.input_data import load_MNIST

"""
Regress on LCA activity
Outputs:
  Score on test data
Inputs:
  train_data: data to fit
  train_labels: training ground-truth (should be 1-hot)
  test_data: data to test fit on
  test_labels: testing ground-truth (should be 1-hot)
  batch_size: size of batch for training
  num_trials: number of batches to train on
"""
def do_regression(train_data, train_labels, test_data, test_labels, sched,
  batch_size=100, num_trials=30000, rand_seed=None):

  (num_neurons, num_trn_examples) = train_data.shape
  num_classes = train_labels.shape[0]
  num_tst_examples = test_data.shape[1]

  if rand_seed:
    tf.set_random_seed(rand_seed)

  global_step = tf.Variable(0, trainable=False, name="global_step")

  x = tf.placeholder(tf.float32,
    shape=[num_neurons, None], name="input_data")
  y = tf.placeholder(tf.float32,
    shape=[num_classes, None], name="input_label")

  w_init = tf.truncated_normal([num_classes, num_neurons], mean=0.0, stddev=1.0,
   dtype=tf.float32, name="w_init")
  w = tf.Variable(w_init, dtype=tf.float32, trainable=True, name="w")
  b = tf.Variable(tf.zeros([num_classes], dtype=tf.float32), trainable=True,
      name="bias")

  y_ = tf.transpose(tf.nn.softmax(tf.transpose(
    tf.matmul(w, x, name="classify")), name="softmax"))

  cross_entropy = -tf.reduce_sum(tf.mul(y,
    tf.log(tf.clip_by_value(y_, 1e-10, 1.0))))

  learning_rates = tf.train.exponential_decay(
    learning_rate=sched["lr"],
    global_step=global_step,
    decay_steps=sched["decay_steps"],
    decay_rate=sched["decay_rate"],
    staircase=sched["staircase"],
    name="annealing_schedule")

  grad_op = tf.train.GradientDescentOptimizer(learning_rates)

  train_step = grad_op.minimize(cross_entropy, global_step=global_step,
    var_list=[w])

  correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  init_op = tf.initialize_all_variables()

  with tf.Session() as sess:
    with tf.device("/cpu:0"):
      sess.run(init_op)

      curr_batch_idx = 0
      for i in range(num_trials):
        data = train_data[:, curr_batch_idx:curr_batch_idx+batch_size]
        labels = train_labels[:, curr_batch_idx:curr_batch_idx+batch_size]

        curr_batch_idx += batch_size
        if curr_batch_idx >= train_data.shape[1]:
          curr_batch_idx = 0

        sess.run(train_step, feed_dict={x:data, y:labels})

      test_accuracy = sess.run(accuracy, feed_dict={x:test_data, y:test_labels})

  return (num_tst_examples * (1.0 - test_accuracy))
