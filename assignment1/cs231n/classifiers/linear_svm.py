import numpy as np
from random import shuffle
#from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0

  if 1:
    # Own solution
    for i in xrange(num_train):
      scores = X[i].dot(W) # [num_train, D] x [D, num_classes]
      correct_class_score = scores[y[i]]
      indicator = (scores-correct_class_score+1.0) > 0 # [num_classes]
      for j in xrange(num_classes):
        if j == y[i]:
          # Correct class.
          dW[:,j] += -np.sum(np.delete(indicator,j))*X[i,:]
          continue
        # indicator same as margin > 0
        dW[:,j] += indicator[j]*X[i,:]
        margin = scores[j] - correct_class_score + 1.0
        if margin > 0:
          loss += margin
  elif 1:
    for i in xrange(num_train):
      scores = X[i].dot(W)
      correct_class_score = scores[y[i]]
      for j in xrange(num_classes):
        if j == y[i]:
          continue
        margin = scores[j] - correct_class_score + 1.0
        if margin > 0:
          loss += margin
          dW[:,j] += X[i].T
          # Subtract again for the correct class
          dW[:,y[i]] += -X[i].T
  else:
    for i in xrange(num_train):
      total_count = 0
      scores = X[i].dot(W)
      correct_class_score = scores[y[i]]
      for j in xrange(num_classes):
        if j == y[i]:
          continue
        margin = scores[j] - correct_class_score + 1 # note delta = 1
        if margin > 0:
          loss += margin
          total_count += 1
          dW[:,j] += X[i]
      dW[:,y[i]] -=  total_count * X[i]

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W

  #############################################################################
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################

  return loss, dW


def svm_loss_vectorized_old(W1, X1, y, reg):
  """
  Loss difference to scalar is non-zero (but small)
  Gradient different is smaller than below (but non-zero)
  """
  delta = 1.0
  W = W1.T
  X = X1.T
  num_train = X.shape[0]

  scores = W.dot(X)
  correct_scores = np.choose(y,scores)
  correct_scores = np.tile(correct_scores,(scores.shape[0],1))

  scores = np.maximum(0,scores-correct_scores+delta)
  loss = np.sum(scores)/num_train
  loss += 0.5 * reg * np.sum(W * W)


  weights = (scores > 0).sum(axis = 0) - 1 # subtract one because correct class score will be delta
  scores[scores > 0] = 1
  tran_scores = scores.T
  tran_scores[range(y.shape[0]),y] = -weights

  dW = tran_scores.T.dot(X.T)
  dW /= num_train
  reg_der = reg * W
  dW = dW +  reg_der

  return loss, dW.T

def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  num_train = X.shape[0]
  num_classes = W.shape[1]

  #############################################################################
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
 # X [num_train, D], W [D, num_classes]
  # print("[num_train, num_classes]: [%d %d]" % (num_train, num_classes))

  scores = X.dot(W) # (num_train, num_classes)
  correct_class_score = scores[np.arange(num_train), y] # (num_train)
  margins = np.maximum(0, scores - correct_class_score[:, np.newaxis] + 1.0)

  # Margins for correct class is zeroed out
  margins[np.arange(num_train), y] = 0
  loss = np.sum(margins) / num_train + 0.5 * reg * np.sum(W * W)

  # This is wrong - contains scores for the first num_classes train images (repeated)
  # correct_class_score = scores[y] # (num_train, num_classes)
  # margins = np.maximum(0, scores - correct_class_score + 1.0)
  # margins[y] = 0
  # loss = np.sum(margins) / num_train + 0.5 * reg * np.sum(W*W)

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################

  if 0:
    coeff_mat = np.zeros((num_train, num_classes))
    coeff_mat[margins > 0] = 1
    # Already zeroed out
    #coeff_mat[range(num_train), list(y)] = 0
    coeff_mat[range(num_train), list(y)] = -np.sum(coeff_mat, axis=1)

    dW = X.T.dot(coeff_mat)
    dW = dW/num_train + reg*W
  else:
    mask = np.zeros(margins.shape,dtype=np.float64)
    mask[margins > 0.0] = 1.0
    mask[np.arange(num_train), y] = - np.sum(mask, axis=1) # subtract incorrect counts
    dW = X.T.dot(mask) / num_train + reg * W

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
