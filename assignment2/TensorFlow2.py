import sys
import tensorflow as tf
import numpy as np
import math
import timeit
import matplotlib.pyplot as plt

useGPU = False
variance_epsilon = 1e-6

if useGPU:
  strDevice = "/gpu:0"
else:
  strDevice = "/cpu:0"

plt.ion()

from cs231n.data_utils import load_CIFAR10

def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=10000):
  """
  Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
  it for the two-layer neural net classifier. These are the same steps as
  we used for the SVM, but condensed to a single function.
  """
  # Load the raw CIFAR-10 data
  cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'
  X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

  # Subsample the data
  mask = range(num_training, num_training + num_validation)
  X_val = X_train[mask]
  y_val = y_train[mask]
  mask = range(num_training)
  X_train = X_train[mask]
  y_train = y_train[mask]
  mask = range(num_test)
  X_test = X_test[mask]
  y_test = y_test[mask]

  # Normalize the data: subtract the mean image
  mean_image = np.mean(X_train, axis=0)
  X_train -= mean_image
  X_val -= mean_image
  X_test -= mean_image

  return X_train, y_train, X_val, y_val, X_test, y_test


# Invoke the above function to get our data.
X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data()
print('Train data shape: ', X_train.shape)
print('Train labels shape: ', y_train.shape)
print('Validation data shape: ', X_val.shape)
print('Validation labels shape: ', y_val.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)

# clear old variables
tf.reset_default_graph()

# setup input (e.g. the data that changes every batch)
# The first dim is None, and gets sets automatically based on batch size fed in
X = tf.placeholder(tf.float32, [None, 32, 32, 3])
y = tf.placeholder(tf.int64, [None])
is_training = tf.placeholder(tf.bool)

def simple_model(X,y):
  # define our weights (e.g. init_two_layer_convnet)

  # setup variables
  Wconv1 = tf.get_variable("Wconv1", shape=[7, 7, 3, 32])
  bconv1 = tf.get_variable("bconv1", shape=[32])
  W1 = tf.get_variable("W1", shape=[5408, 10])
  b1 = tf.get_variable("b1", shape=[10])

  # define our graph (e.g. two_layer_convnet)
  a1 = tf.nn.conv2d(X, Wconv1, strides=[1,2,2,1], padding='VALID') + bconv1
  h1 = tf.nn.relu(a1)
  h1_flat = tf.reshape(h1,[-1,5408])
  y_out = tf.matmul(h1_flat,W1) + b1
  return y_out

y_out = simple_model(X,y)

# define our loss
total_loss = tf.losses.hinge_loss(tf.one_hot(y,10),logits=y_out)
mean_loss = tf.reduce_mean(total_loss)

# define our optimizer
optimizer = tf.train.AdamOptimizer(5e-4) # select optimizer and set learning rate
train_step = optimizer.minimize(mean_loss)

def run_model(session, predict, loss_val, Xd, yd,
            epochs=1, batch_size=64, print_every=100,
            training=None, plot_losses=False):
  # have tensorflow compute accuracy
  correct_prediction = tf.equal(tf.argmax(predict,1), y)
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  # shuffle indicies
  train_indicies = np.arange(Xd.shape[0])
  np.random.shuffle(train_indicies)

  training_now = training is not None

  # setting up variables we want to compute (and optimizing)
  # if we have a training function, add that to things we compute
  variables = [mean_loss,correct_prediction,accuracy]
  if training_now:
    variables[-1] = training

  # counter
  iter_cnt = 0
  for e in range(epochs):
    # keep track of losses and accuracy
    correct = 0
    losses = []
    # make sure we iterate over the dataset once
    for i in range(int(math.ceil(Xd.shape[0]/batch_size))):
      # generate indicies for the batch
      start_idx = (i*batch_size)%Xd.shape[0]
      idx = train_indicies[start_idx:start_idx+batch_size]

      # create a feed dictionary for this batch
      feed_dict = {X: Xd[idx,:],
                   y: yd[idx],
                   is_training: training_now }
      # get batch size
      actual_batch_size = yd[idx].shape[0]

      # have tensorflow compute loss and correct predictions
      # and (if given) perform a training step
      loss, corr, _ = session.run(variables,feed_dict=feed_dict)

      # aggregate performance stats
      losses.append(loss*actual_batch_size)
      correct += np.sum(corr)

      # print every now and then
      if training_now and (iter_cnt % print_every) == 0:
        print("Iteration {0}: with minibatch training loss = {1:.3g} and accuracy of {2:.2g}"\
              .format(iter_cnt,loss,float(np.sum(corr))/actual_batch_size))
      iter_cnt += 1
    total_correct = float(correct)/Xd.shape[0]
    total_loss = float(np.sum(losses))/Xd.shape[0]
    print("Epoch {2}, Overall loss = {0:.3g} and accuracy of {1:.3g}"\
          .format(total_loss,total_correct,e+1))
    if plot_losses:
      plt.plot(losses)
      plt.grid(True)
      plt.title('Epoch {} Loss'.format(e+1))
      plt.xlabel('minibatch number')
      plt.ylabel('minibatch loss')
      plt.show()
  return total_loss,total_correct

with tf.Session() as sess:
  with tf.device(strDevice):
    sess.run(tf.global_variables_initializer())
    print('Training')
    run_model(sess,y_out,mean_loss,X_train,y_train,1,64,100,train_step,True)
    print('Validation')
    run_model(sess,y_out,mean_loss,X_val,y_val,1,64)

# clear old variables
tf.reset_default_graph()

# define our input (e.g. the data that changes every batch)
# The first dim is None, and gets sets automatically based on batch size fed in
X = tf.placeholder(tf.float32, [None, 32, 32, 3])
y = tf.placeholder(tf.int64, [None])
is_training = tf.placeholder(tf.bool)

# define model
def complex_model(X,y,is_training):
  with tf.name_scope("ConvLayer"):
    W_conv1 = tf.get_variable("W_conv1", shape=[7, 7, 3, 32])
    b_conv1 = tf.get_variable("b_conv1", shape=[32])

    h_conv = tf.nn.conv2d(X, W_conv1, strides=[1, 1, 1, 1], padding="VALID") + b_conv1
    h_conv1 = tf.nn.relu(h_conv)

  if useGPU:
    with tf.name_scope("BatchNorm1"):
      h_bn0 = tf.layers.batch_normalization(h_conv1, axis=1, training=is_training)
  else:
    with tf.name_scope("BatchNorm1"):
      sconv = tf.get_variable("sc", shape=[26, 26, 32]) # bn scale param
      oconv = tf.get_variable("oc", shape=[26, 26, 32]) # bn offset param
      mc1, vc1 = tf.nn.moments(h_conv1, axes=[0], keep_dims=False)
      h_bn0 = tf.nn.batch_normalization(h_conv1, mc1, vc1, oconv, sconv, variance_epsilon)

  with tf.name_scope("max-pool"):
    h_pool = tf.nn.max_pool(h_bn0, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
  with tf.name_scope("FullyConnected"):
    h_reshaped = tf.reshape(h_pool, [-1, 13*13*32])
    W1 = tf.get_variable("W1", shape=[13*13*32, 1024])
    b1 = tf.get_variable("b1", shape=[1024])

    h_fc = tf.matmul(h_reshaped, W1) + b1
    h_fc1 = tf.nn.relu(h_fc)

  if useGPU:
    with tf.name_scope("BatchNorm2"):
      h_bn1 = tf.layers.batch_normalization(h_fc1, axis=1, training=is_training)
  else:
    with tf.name_scope("BatchNorm2"):
      s1 = tf.get_variable("s1", shape=[1024])
      o1 = tf.get_variable("o1", shape=[1024])
      m1, v1 = tf.nn.moments(h_fc1, axes=[0], keep_dims=False)
      h_bn1 = tf.nn.batch_normalization(h_fc1, m1, v1, o1, s1, variance_epsilon)

  with tf.name_scope("FullyConnectedOut"):
    W2 = tf.get_variable("W2", shape=[1024, 10])
    b2 = tf.get_variable("b2", shape=[10])

    scores = tf.matmul(h_bn1, W2) + b2

  return scores

if 0:
  y_out = complex_model(X,y,is_training)

  # Now we're going to feed a random batch into the model
  # and make sure the output is the right size
  x = np.random.randn(64, 32, 32, 3)
  with tf.Session() as sess:
    with tf.device(strDevice): #"/cpu:0" or "/gpu:0"
      tf.global_variables_initializer().run()

      ans = sess.run(y_out,feed_dict={X:x,is_training:True})
      timeit.Timer('sess.run(y_out,feed_dict={X:x,is_training:True})')
      print(ans.shape)
      print(np.array_equal(ans.shape, np.array([64, 10])))

  try:
    with tf.Session() as sess:
      with tf.device(strDevice) as dev: #"/cpu:0" or "/gpu:0"
        tf.global_variables_initializer().run()
        ans = sess.run(y_out,feed_dict={X:x,is_training:True})
        timeit.Timer('sess.run(y_out,feed_dict={X:x,is_training:True})')
  except tf.errors.InvalidArgumentError:
    print("no gpu found, please use Google Cloud if you want GPU acceleration")
    # rebuild the graph
    # trying to start a GPU throws an exception
    # and also trashes the original graph
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, [None, 32, 32, 3])
    y = tf.placeholder(tf.int64, [None])
    is_training = tf.placeholder(tf.bool)
    y_out = complex_model(X,y,is_training)

  # Inputs
  #     y_out: is what your model computes
  #     y: is your TensorFlow variable with label information
  # Outputs
  #    mean_loss: a TensorFlow variable (scalar) with numerical loss
  #    optimizer: a TensorFlow optimizer
  # This should be ~3 lines of code!
  mean_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(y, 10), logits=y_out))
  optimizer = tf.train.RMSPropOptimizer(1e-3)

  # batch normalization in tensorflow requires this extra dependency
  extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  with tf.control_dependencies(extra_update_ops):
    train_step = optimizer.minimize(mean_loss)

  sess = tf.Session()

  sess.run(tf.global_variables_initializer())
  print('Training')
  run_model(sess,y_out,mean_loss,X_train,y_train,1,64,100,train_step)

  print('Validation')
  run_model(sess,y_out,mean_loss,X_val,y_val,1,64)

tf.reset_default_graph()
# Feel free to play with this cell
def my_model(X,y,is_training):
  with tf.name_scope("ConvLayer1"):
    W_conv1 = tf.get_variable("W_conv1", shape=[5, 5, 3, 32])
    b_conv1 = tf.get_variable("b_conv1", shape=[32])
    a1 = tf.nn.conv2d(X, W_conv1, strides=[1, 1, 1, 1], padding="SAME") + b_conv1
    h_conv1 = tf.nn.relu(a1)

  if useGPU:
    with tf.name_scope("BatchNorm1"):
      h_bn1 = tf.layers.batch_normalization(h_conv1, axis=1, training=is_training)
  else:
    with tf.name_scope("BatchNorm1"):
      sconv = tf.get_variable("sc", shape=[32, 32, 32]) # bn scale param
      oconv = tf.get_variable("oc", shape=[32, 32, 32]) # bn offset param
      mc1, vc1 = tf.nn.moments(h_conv1, axes=[0], keep_dims=False)
      h_bn1 = tf.nn.batch_normalization(h_conv1, mc1, vc1, oconv, sconv, variance_epsilon)

  with tf.name_scope("maxpool"):
    h_pool = tf.nn.max_pool(h_bn1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")

  with tf.name_scope("ConvLayer2"):
    W_conv2 = tf.get_variable("W_conv2", shape=[5, 5, 32, 64])
    b_conv2 = tf.get_variable("b_conv2", shape=[64])

    a2 = tf.nn.conv2d(h_pool, W_conv2, strides=[1, 1, 1, 1], padding="SAME") + b_conv2
    h_conv2 = tf.nn.relu(a2)

  if useGPU:
    with tf.name_scope("BatchNorm2"):
      h_bn2 = tf.layers.batch_normalization(h_conv2, axis=1, training=is_training)
  else:
    with tf.name_scope("BatchNorm2"):
      sconv2 = tf.get_variable("sc2", shape=[16, 16, 64]) # bn scale param
      oconv2 = tf.get_variable("oc2", shape=[16, 16, 64]) # bn offset param
      mc2, vc2 = tf.nn.moments(h_conv2, axes=[0], keep_dims=False)
      h_bn2 = tf.nn.batch_normalization(h_conv2, mc2, vc2, oconv2, sconv2, variance_epsilon)

  with tf.name_scope("FullyConnected"):
    h_reshaped = tf.reshape(h_bn2, [-1, 16*16*64])

    W1 = tf.get_variable("W1", shape=[16*16*64, 1024])
    b1 = tf.get_variable("b1", shape=[1024])

    a3 = tf.matmul(h_reshaped, W1) + b1
    h_fc1 = tf.nn.relu(a3)

  with tf.name_scope("FullyConnectedOut"):
    W2 = tf.get_variable("W2", shape=[1024, 10])
    b2 = tf.get_variable("b2", shape=[10])

    scores = tf.matmul(h_fc1, W2) + b2

  return scores

tf.reset_default_graph()

X = tf.placeholder(tf.float32, [None, 32, 32, 3])
y = tf.placeholder(tf.int64, [None])
is_training = tf.placeholder(tf.bool)

y_out = my_model(X,y,is_training)
mean_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(y, 10), logits=y_out))
optimizer = tf.train.AdamOptimizer(5e-4)

# TODO: Use tf.train.AdadeltaOptimizer

# batch normalization in tensorflow requires this extra dependency
extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(extra_update_ops):
  train_step = optimizer.minimize(mean_loss)

# Feel free to play with this cell
# This default code creates a session
# and trains your model for 10 epochs
# then prints the validation set accuracy
sess = tf.Session()

sess.run(tf.global_variables_initializer())
print('Training')
run_model(sess,y_out,mean_loss,X_train,y_train,10,64,100,train_step,True)
print('Validation')
run_model(sess,y_out,mean_loss,X_val,y_val,1,64)

# Test your model here, and make sure
# the output of this cell is the accuracy
# of your best model on the training and val sets
# We're looking for >= 70% accuracy on Validation
print('Test')
run_model(sess,y_out,mean_loss,X_test,y_test,1,64)

# It is saved - 0.679 on test set
#saver = tf.train.Saver()
#save_path = saver.save(sess, "./TensorFlow2.ckpt")

# Local variables: #
# tab-width: 2 #
# python-indent: 2 #
# indent-tabs-mode: nil #
# End: #
