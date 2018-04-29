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

def my_model(X,y,is_training):
  with tf.name_scope("ConvLayer1"):
    W_conv1 = tf.get_variable("W_conv1", shape=[5, 5, 3, 16])
    b_conv1 = tf.get_variable("b_conv1", shape=[16])
    a1 = tf.nn.conv2d(X, W_conv1, strides=[1, 1, 1, 1], padding="SAME") + b_conv1
    h_conv1 = tf.nn.relu(a1)

  if useGPU:
    with tf.name_scope("BatchNorm1"):
      h_bn1 = tf.layers.batch_normalization(h_conv1, axis=1, training=is_training)
  else:
    with tf.name_scope("BatchNorm1"):
      sconv = tf.get_variable("sc", shape=[32, 32, 16]) # bn scale param
      oconv = tf.get_variable("oc", shape=[32, 32, 16]) # bn offset param
      mc1, vc1 = tf.nn.moments(h_conv1, axes=[0], keep_dims=False)
      h_bn1 = tf.nn.batch_normalization(h_conv1, mc1, vc1, oconv, sconv, variance_epsilon)

  with tf.name_scope("maxpool"):
    h_pool = tf.nn.max_pool(h_bn1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")

  with tf.name_scope("ConvLayer2"):
    W_conv2 = tf.get_variable("W_conv2", shape=[5, 5, 16, 64])
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



# learning_rate=1., rho=0.95, epsilon=1e-6

# layer_defs = [];
# layer_defs.push({type:'input', out_sx:32, out_sy:32, out_depth:3});
# layer_defs.push({type:'conv', sx:5, filters:16, stride:1, pad:2, activation:'relu'});
# layer_defs.push({type:'pool', sx:2, stride:2});
# layer_defs.push({type:'conv', sx:5, filters:20, stride:1, pad:2, activation:'relu'});
# layer_defs.push({type:'pool', sx:2, stride:2});
# layer_defs.push({type:'conv', sx:5, filters:20, stride:1, pad:2, activation:'relu'});
# layer_defs.push({type:'pool', sx:2, stride:2});
# layer_defs.push({type:'softmax', num_classes:10});
#
# net = new convnetjs.Net();
# net.makeLayers(layer_defs);
#
# trainer = new convnetjs.SGDTrainer(net, {method:'adadelta', batch_size:4, l2_decay:0.0001});
