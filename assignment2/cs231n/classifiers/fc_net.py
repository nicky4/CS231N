import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network with ReLU nonlinearity and
  softmax loss that uses a modular layer design. We assume an input dimension
  of D, a hidden dimension of H, and perform classification over C classes.
  
  The architecure should be affine - relu - affine - softmax.

  Note that this class does not implement gradient descent; instead, it
  will interact with a separate Solver object that is responsible for running
  optimization.

  The learnable parameters of the model are stored in the dictionary
  self.params that maps parameter names to numpy arrays.
  """
  
  def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
               weight_scale=1e-3, reg=0.0):
    """
    Initialize a new network.

    Inputs:
    - input_dim: An integer giving the size of the input
    - hidden_dim: An integer giving the size of the hidden layer
    - num_classes: An integer giving the number of classes to classify
    - dropout: Scalar between 0 and 1 giving dropout strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - reg: Scalar giving L2 regularization strength.
    """
    self.params = {}
    self.reg = reg
    
    ############################################################################
    # TODO: Initialize the weights and biases of the two-layer net. Weights    #
    # should be initialized from a Gaussian with standard deviation equal to   #
    # weight_scale, and biases should be initialized to zero. All weights and  #
    # biases should be stored in the dictionary self.params, with first layer  #
    # weights and biases using the keys 'W1' and 'b1' and second layer weights #
    # and biases using the keys 'W2' and 'b2'.                                 #
    ############################################################################
    self.params['W1'] = weight_scale * np.random.randn(input_dim, hidden_dim)
    self.params['W2'] = weight_scale * np.random.randn(hidden_dim, num_classes)
    self.params['b1'] = np.zeros(hidden_dim)
    self.params['b2'] = np.zeros(num_classes)
    pass
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################


  def loss(self, X, y=None):
    """
    Compute loss and gradient for a minibatch of data.

    Inputs:
    - X: Array of input data of shape (N, d_1, ..., d_k)
    - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

    Returns:
    If y is None, then run a test-time forward pass of the model and return:
    - scores: Array of shape (N, C) giving classification scores, where
      scores[i, c] is the classification score for X[i] and class c.

    If y is not None, then run a training-time forward and backward pass and
    return a tuple of:
    - loss: Scalar value giving the loss
    - grads: Dictionary with the same keys as self.params, mapping parameter
      names to gradients of the loss with respect to those parameters.
    """  
    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the two-layer net, computing the    #
    # class scores for X and storing them in the scores variable.              #
    ############################################################################
#    l1 = np.maximum(X.dot(self.params['W1']),0)+self.params['b1']
#    l2 = l1.dot(self.params['W2'])+self.params['b2']
    l1, cache = affine_relu_forward(X, self.params['W1'], self.params['b1'])
    l2, cache= affine_forward(l1, self.params['W2'], self.params['b2'])
    scores = l2
    pass
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # If y is None then we are in test mode so just return scores
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the two-layer net. Store the loss  #
    # in the loss variable and gradients in the grads dictionary. Compute data #
    # loss using softmax, and make sure that grads[k] holds the gradients for  #
    # self.params[k]. Don't forget to add L2 regularization!                   #
    #                                                                          #
    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization includes a factor #
    # of 0.5 to simplify the expression for the gradient.                      #
    ############################################################################
    loss, dscore = softmax_loss(scores, y)
    
    grads['W2'] = l1.T.dot(dscore)
    grads['W2'] += self.reg * self.params['W2']
    grads['b2'] = np.sum(dscore, axis=0)
    
    dhidden = np.dot(dscore, self.params['W2'].T)
    grads['b1'] = np.sum(dhidden, axis=0)
    dhidden[l1 <= 0] = 0
    
    grads['W1'] = np.reshape(X,[X.shape[0],self.params['W1'].shape[0]]).T.dot(dhidden)
    grads['W1'] += self.reg * self.params['W1']
    loss += 0.5 * self.reg * (np.sum(self.params['W1'] ** 2) + np.sum(self.params['W2'] ** 2))
    pass
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads


class FullyConnectedNet(object):
  """
  A fully-connected neural network with an arbitrary number of hidden layers,
  ReLU nonlinearities, and a softmax loss function. This will also implement
  dropout and batch normalization as options. For a network with L layers,
  the architecture will be
  
  {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax
  
  where batch normalization and dropout are optional, and the {...} block is
  repeated L - 1 times.
  
  Similar to the TwoLayerNet above, learnable parameters are stored in the
  self.params dictionary and will be learned using the Solver class.
  """

  def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
               dropout=0, use_batchnorm=False, reg=0.0,
               weight_scale=1e-2, dtype=np.float32, seed=None):
    """
    Initialize a new FullyConnectedNet.
    
    Inputs:
    - hidden_dims: A list of integers giving the size of each hidden layer.
    - input_dim: An integer giving the size of the input.
    - num_classes: An integer giving the number of classes to classify.
    - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
      the network should not use dropout at all.
    - use_batchnorm: Whether or not the network should use batch normalization.
    - reg: Scalar giving L2 regularization strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - dtype: A numpy datatype object; all computations will be performed using
      this datatype. float32 is faster but less accurate, so you should use
      float64 for numeric gradient checking.
    - seed: If not None, then pass this random seed to the dropout layers. This
      will make the dropout layers deteriminstic so we can gradient check the
      model.
    """
    self.use_batchnorm = use_batchnorm
    self.use_dropout = dropout > 0
    self.reg = reg
    self.num_layers = 1 + len(hidden_dims)
    self.dtype = dtype
    self.params = {}

    ############################################################################
    # TODO: Initialize the parameters of the network, storing all values in    #
    # the self.params dictionary. Store weights and biases for the first layer #
    # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
    # initialized from a normal distribution with standard deviation equal to  #
    # weight_scale and biases should be initialized to zero.                   #
    #                                                                          #
    # When using batch normalization, store scale and shift parameters for the #
    # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
    # beta2, etc. Scale parameters should be initialized to one and shift      #
    # parameters should be initialized to zero.                                #
    ############################################################################

    for i in np.arange(self.num_layers-1):
        if i == 0:
            self.params['W1'] = weight_scale * np.random.randn(input_dim, hidden_dims[0])
            self.params['b1'] = np.zeros(hidden_dims[0])
            if self.use_batchnorm:
                self.params['gamma1'] = np.random.randn(hidden_dims[0])
                self.params['beta1'] = np.random.randn(hidden_dims[0])
        else:
            self.params['W%d' % float(i+1)] = weight_scale * np.random.randn(hidden_dims[i-1], hidden_dims[i])
            self.params['b%d' % float(i+1)] = np.zeros(hidden_dims[i])
            if self.use_batchnorm:
                self.params['gamma%d' % float(i+1)] = np.random.randn(hidden_dims[i])
                self.params['beta%d' % float(i+1)] = np.random.randn(hidden_dims[i])
    self.params['W%d' % float(self.num_layers)] = weight_scale * np.random.randn(hidden_dims[self.num_layers-2], num_classes)
    self.params['b%d' % float(self.num_layers)] = np.zeros(num_classes)
    pass
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # When using dropout we need to pass a dropout_param dictionary to each
    # dropout layer so that the layer knows the dropout probability and the mode
    # (train / test). You can pass the same dropout_param to each dropout layer.
    self.dropout_param = {}
    if self.use_dropout:
      self.dropout_param = {'mode': 'train', 'p': dropout}
      if seed is not None:
        self.dropout_param['seed'] = seed
    
    # With batch normalization we need to keep track of running means and
    # variances, so we need to pass a special bn_param object to each batch
    # normalization layer. You should pass self.bn_params[0] to the forward pass
    # of the first batch normalization layer, self.bn_params[1] to the forward
    # pass of the second batch normalization layer, etc.
    self.bn_params = []
    if self.use_batchnorm:
      self.bn_params = [{'mode': 'train'} for i in xrange(self.num_layers - 1)]
    
    # Cast all parameters to the correct datatype
    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)


  def loss(self, X, y=None):
    """
    Compute loss and gradient for the fully-connected net.

    Input / output: Same as TwoLayerNet above.
    """
    X = X.astype(self.dtype)
    mode = 'test' if y is None else 'train'

    # Set train/test mode for batchnorm params and dropout param since they
    # behave differently during training and testing.
    if self.dropout_param is not None:
      self.dropout_param['mode'] = mode   
    if self.use_batchnorm:
      for bn_param in self.bn_params:
        bn_param[mode] = mode

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the fully-connected net, computing  #
    # the class scores for X and storing them in the scores variable.          #
    #                                                                          #
    # When using dropout, you'll need to pass self.dropout_param to each       #
    # dropout forward pass.                                                    #
    #                                                                          #
    # When using batch normalization, you'll need to pass self.bn_params[0] to #
    # the forward pass for the first batch normalization layer, pass           #
    # self.bn_params[1] to the forward pass for the second batch normalization #
    # layer, etc.                                                              #
    ############################################################################
    l = []
    cache_list = []
    l.append(X.reshape(X.shape[0], np.product(X.shape[1:])))
    
    # (Affine - Relu) * (L - 1)
    for i in np.arange(self.num_layers-1):
        cache = []
        if i == 0:
            l_temp, fc_cache = affine_forward(X, self.params['W1'], self.params['b1'])
            cache.append(fc_cache)
            if (self.use_batchnorm):
                l_temp, bn_cache = batchnorm_forward(l_temp, self.params['gamma1'], self.params['beta1'], self.bn_params[0])
                cache.append(bn_cache)
            l_temp, relu_cache = relu_forward(l_temp)
            cache.append(relu_cache)
            if (self.use_dropout):
                l_temp, dp_cache = dropout_forward(l_temp, self.dropout_param)
                cache.append(dp_cache)

            l.append(l_temp)
            X = cache[0][0]
            cache_list.append(cache)
        else:
            l_temp, fc_cache = affine_forward(l[i], self.params['W%d' % float(i+1)], self.params['b%d' % float(i+1)])
            cache.append(fc_cache)
            
            if (self.use_batchnorm):
                l_temp, bn_cache = batchnorm_forward(l_temp, self.params['gamma%d' % float(i+1)], self.params['beta%d' % float(i+1)], self.bn_params[i])
                cache.append(bn_cache)
            
            l_temp, relu_cache = relu_forward(l_temp)
            cache.append(relu_cache)
            if (self.use_dropout):
                l_temp, dp_cache = dropout_forward(l_temp, self.dropout_param)
                cache.append(dp_cache)

            cache_list.append(cache)
            l.append(l_temp)
    
    # Affine
    l_temp, cache = affine_forward(l[self.num_layers-1], self.params['W%d' % float(self.num_layers)], self.params['b%d' % float(self.num_layers)])
    cache_list.append([cache,[]])
    l.append(l_temp)
    scores = l_temp
        
    pass
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # If test mode return early
    if mode == 'test':
      return scores

    loss, grads = 0.0, {}
    ############################################################################
    # TODO: Implement the backward pass for the fully-connected net. Store the #
    # loss in the loss variable and gradients in the grads dictionary. Compute #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    #                                                                          #
    # When using batch normalization, you don't need to regularize the scale   #
    # and shift parameters.                                                    #
    #                                                                          #
    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization includes a factor #
    # of 0.5 to simplify the expression for the gradient.                      #
    ############################################################################
    loss, dscore = softmax_loss(scores, y)

#    for i in np.arange(self.num_layers, 0, -1):
#        if i == self.num_layers:
#            loss += 0.5 * self.reg * np.sum(self.params['W%d' % float(i)] ** 2) # L2 regularization
#            grads['W%d' % float(i)] = l[i-1].T.dot(dscore)
#            grads['W%d' % float(i)] += self.reg * self.params['W%d' % float(i)]
#            grads['b%d' % float(i)] = np.sum(dscore, axis = 0)
#        else:
#            loss += 0.5 * self.reg * np.sum(self.params['W%d' % float(i)] ** 2) # L2 regularization
#            dhidden = np.dot(dscore, self.params['W%d' % float(i+1)].T)
#            dhidden[l[i] <= 0] = 0
#    
#            grads['W%d' % float(i)] = l[i-1].T.dot(dhidden)
#            grads['W%d' % float(i)] += self.reg * self.params['W%d' % float(i)]
#            dscore = dhidden
#            grads['b%d' % float(i)] = np.sum(dhidden, axis = 0)


    for i in np.arange(self.num_layers, 0, -1):
            loss += 0.5 * self.reg * np.sum(self.params['W%d' % float(i)] ** 2) # L2 regularization
            if i == self.num_layers:
                dx, dw, db = affine_backward(dscore, cache_list[i-1][0])
                grads['W%d' % float(i)] = dw
                grads['W%d' % float(i)] += self.reg * self.params['W%d' % float(i)]
                grads['b%d' % float(i)] = db
                dscore = dx
                continue
            if self.use_dropout:
                dx = dropout_backward(dscore, cache_list[i-1][-1])
                dx = relu_backward(dx, cache_list[i-1][-2])
            else:
                dx = relu_backward(dscore, cache_list[i-1][-1])
            if self.use_batchnorm:
                dx, dgamma, dbeta = batchnorm_backward(dx, cache_list[i-1][1])
                grads['gamma%d' % float(i)] = dgamma
                grads['beta%d' % float(i)] = dbeta
    
            dx, dw, db = affine_backward(dx, cache_list[i-1][0])

            grads['W%d' % float(i)] = dw
            grads['W%d' % float(i)] += self.reg * self.params['W%d' % float(i)]
            grads['b%d' % float(i)] = db
            dscore = dx

#    grads['W2'] = l1.T.dot(dscore)
#    grads['W2'] += self.reg * self.params['W2']
#    grads['b2'] = np.sum(dscore, axis=0)
#    
#    dhidden = np.dot(dscore, self.params['W2'].T)
#    grads['b1'] = np.sum(dhidden, axis=0)
#    dhidden[l1 <= 0] = 0
#    
#    grads['W1'] = X.T.dot(dhidden)
#    grads['W1'] += self.reg * self.params['W1']

    pass
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads