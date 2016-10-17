import numpy as np


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
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        #change here
        loss += margin
        dW[:,j]+=X[i]
        dW[:,y[i]]-=X[i]

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  dW /= num_train
  dW+= reg*W
  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  delta=1.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################

  N = X.shape[0]
  Y_predict=X.dot(W)
  margin = Y_predict - Y_predict[np.arange(N), y].reshape(-1, 1) + delta
  margin[np.arange(N), y] = 0
  wrong_p=margin>0
  margin=margin[wrong_p]
  loss = 1.0 / N * np.sum(margin) + reg * np.sum(W ** 2) * 0.5

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  # dW=X[wrong_p[: ]]
  N, C = Y_predict.shape
  D = X.shape[1]

  # Y_bcst=Y_predict.reshape(N,1,C)
  Wrong_p_bcst=wrong_p.reshape(N,1,C)
  X_bcst=X.reshape(N,D,1)

  d_W1=X_bcst*Wrong_p_bcst

  dW1=d_W1.sum(axis=0)
  # yy=y.reshape(N,1,1)
  # d_w2=Wrong_p_bcst*yy
  wrongnum = wrong_p.sum(axis=1).reshape(-1, 1)
  util_v = np.zeros([N, C])
  util_v[np.arange(N), y] = 1
  wrongnum = (wrongnum * util_v).reshape(N, 1, C)
  dW2=np.zeros_like(dW1)
  dW2 -= np.sum(wrongnum * X_bcst, axis=0)
  dW3 = reg * W
  dW = (dW1 + dW2) / (N + 0.0) + dW3
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
