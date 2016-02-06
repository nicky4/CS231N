import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *

class DeepNet(object):
  """
  A n-layer convolutional network with the following architecture:
  
  (conv - batchNorm - relu - dropout - 2x2 max pool) x num_conv - (conv - relu) - affine x M - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_conv=2, num_affine=2, num_filters=32, filter_size=7,
               hidden_dim=[100], num_classes=10, weight_scale=1e-3, reg=0.0, dropout=0, use_batchnorm=False,
               dtype=np.float32, seed = None):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.use_batchnorm = use_batchnorm
    self.use_dropout = dropout > 0
    
    self.num_conv = num_conv
    self.num_affine = num_affine
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    C,H,W = input_dim
    
#   conv-relu-pool x N && conv-relu parameters
    for i in np.arange(1, num_conv+2):
      self.params['W%d' % float(i)] = weight_scale * np.random.randn(num_filters,C,filter_size,filter_size)
      self.params['b%d' % float(i)] = np.zeros(num_filters)
      C = num_filters

    # affine parameters
    if num_affine == 1:
        self.params['W%d' % float(num_conv+2)] = weight_scale * np.random.randn(num_filters * H * W / ((2 * 2) ** num_conv), num_classes)
        self.params['b%d' % float(num_conv+2)] = np.zeros(num_classes)
    else:
        for i in np.arange(num_conv+2, num_conv+2+num_affine):
            if i == num_conv+1+num_affine:
                self.params['W%d' % float(i)] = weight_scale * np.random.randn(hidden_dim[-1], num_classes)
                self.params['b%d' % float(i)] = np.zeros(num_classes)
            elif i == num_conv+2:
              self.params['W%d' % float(i)] = weight_scale * np.random.randn(num_filters * H * W / ((2 * 2) ** num_conv), hidden_dim[i - num_conv - 2])
              self.params['b%d' % float(i)] = np.zeros(hidden_dim[i - num_conv - 2])
            else:
              self.params['W%d' % float(i)] = weight_scale * np.random.randn(hidden_dim[i - num_conv - 2], hidden_dim[i - num_conv - 2])
              self.params['b%d' % float(i)] = np.zeros(hidden_dim[i - num_conv - 2])

    pass
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

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
      self.bn_params = [{'mode': 'train'} for i in xrange(num_conv)]

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the deep convolutional network.
    
    (conv - relu - 2x2 max pool) x num_conv - (conv - relu) - affine x M - softmax
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """

    # pass conv_param to the forward pass for the convolutional layer
    filter_size = self.params['W1'].shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None

    # forward pass for the deep convnet
    num_conv = self.num_conv
    num_affine = self.num_affine
    cache_list = []
    out = X
    
    # conv-relu-pool x num_conv
    for i in np.arange(1, num_conv+1):
      W, b = self.params['W%d' % float(i)], self.params['b%d' % float(i)]
      out, cache = conv_relu_pool_forward(out, W, b, conv_param, pool_param)
      cache_list.append(cache)

    # conv - relu
    W, b = self.params['W%d' % float(num_conv+1)], self.params['b%d' % float(num_conv + 1)]
    out, cache = conv_relu_forward(out, W, b, conv_param)
    cache_list.append(cache)

    # affine
    for i in np.arange(num_conv+2, num_conv + 2 + num_affine):
      W, b = self.params['W%d' % float(i)], self.params['b%d' % float(i)]
      out, cache = affine_forward(out, W, b)
      cache_list.append(cache)
    
    scores = out


    pass
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    loss, dscore = softmax_loss(scores, y)
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    # affine backwards
    dout = dscore
    for i in np.arange(num_conv + 1 + num_affine, num_conv + 1, -1):
      loss += 0.5 * self.reg * (np.sum(self.params['W%d' % float(i)] ** 2))
      dout, dw, db = affine_backward(dout, cache_list[i-1])
      grads['W%d' % float(i)] = dw + self.reg * self.params['W%d' % float(i)]
      grads['b%d' % float(i)] = db
    
    # conv-relu backwards
    loss += 0.5 * self.reg * (np.sum(self.params['W%d' % float(num_conv + 1)] ** 2))
    dout, dw, db = conv_relu_backward(dout, cache_list[num_conv])
    grads['W%d' % float(num_conv+1)] = dw + self.reg * self.params['W%d' % float(num_conv+1)]
    grads['b%d' % float(num_conv+1)] = db
    
    # conv-relu-pool backwards
    for i in np.arange(num_conv-1, -1, -1):
      loss += 0.5 * self.reg * (np.sum(self.params['W%d' % float(i+1)] ** 2))
      dout, dw, db = conv_relu_pool_backward(dout, cache_list[i])
      grads['W%d' % float(i+1)] = dw + self.reg * self.params['W%d' % float(i+1)]
      grads['b%d' % float(i+1)] = db

    pass
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  
  
pass

