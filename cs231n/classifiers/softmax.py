import numpy as np


def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

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
  # Initialize the loss and gradient to zero.
  N, D = X.shape
  C = W.shape[1]
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
  for xi, yi in zip(X, y):
    Y = xi.dot(W)
    denominator = np.sum(np.exp(Y))
    loss += np.log(np.exp(Y[yi]) / denominator)
    dW[:, yi] -= xi
    dW += 1 / denominator * xi.reshape(-1, 1) * np.exp(Y).reshape(1, -1)
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  loss /= N
  loss += reg * np.sum(W ** 2)
  dW /= N
  dW += 2 * W
  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  N, D = X.shape
  Y = X.dot(W)
  C = Y.max(axis=1)
  Y -= C.reshape(-1, 1)
  score = np.exp(Y)
  correct_class = score[np.arange(N), y]
  N_sum = np.sum(score, axis=1)
  llh = correct_class / N_sum
  loss += np.sum(llh)
  loss / + N
  loss += reg * W ** 2
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
  dW += (1 / Y.sum(axis=1).reshape(N, 1, 1) * X.reshape(N, 1, D) * Y.reshape(N, C, 1)).np.sum(axis=1)
  dW /= N
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

