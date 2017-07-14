import tensorflow as tf
from tensorflow.python.ops.rnn_cell_impl import _RNNCell as RNNCell
from tensorflow.python.ops.math_ops import tanh
from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.contrib.rnn import LSTMStateTuple

class LSTMCell(RNNCell):
    def __init__(self, num_units, num_proj=None, activation=tanh, reuse=None):
            self._num_units = num_units
            self._num_proj = num_proj
            self._activation = activation
            self._reuse = reuse

            if self._num_proj:
                self._state_size = (self._num_units, self._num_proj)
                self._output_size = self._num_proj
            else:
                self._state_size = (self._num_units, self._num_units)
                self._output_size = self._num_units

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._output_size

    def zero_state(self, batch_size, dtype):
       state_size = self.state_size
       return tuple([tf.zeros([batch_size, s], dtype=dtype) for s in state_size])

    def __call__(self, inputs, state, scope=None):
            c, h = state
            dtype = inputs.dtype
            input_size = inputs.get_shape()[1]
            batch_size = inputs.get_shape()[0]

            #block input
            Wz = tf.get_variable("Wz", [input_size, 
                                    self._num_units], dtype=dtype)
            Rz = tf.get_variable("Rz", [self._output_size, 
                                    self._num_units], dtype=dtype)
            bz = tf.get_variable("bz", [1,  self._num_units], 
                                    dtype=dtype)
            z = self._activation(tf.matmul(inputs, Wz) + tf.matmul(h, Rz) + bz)

            #input gate
            Wi = tf.get_variable("Wi", [input_size, self._num_units], dtype=dtype)
            Ri = tf.get_variable("Ri", [self._output_size, self._num_units], dtype=dtype)
            Pi = tf.get_variable("Pi", [batch_size, self._num_units], dtype=dtype)
            bi = tf.get_variable("bi", [1,  self._num_units], dtype=dtype)
            input_gate = tf.sigmoid(tf.matmul(inputs, Wi) + tf.matmul(h, Ri) + Pi*c + bi)
            
            #forget gate
            Wf = tf.get_variable("Wf", [input_size, self._num_units], dtype=dtype)
            Rf = tf.get_variable("Rf", [self._output_size, self._num_units], dtype=dtype)
            Pf = tf.get_variable("Pf", [batch_size, self._num_units], dtype=dtype)
            bf = tf.get_variable("bf", [1,  self._num_units], dtype=dtype)
            forget_gate = tf.sigmoid(tf.matmul(inputs, Wf) + tf.matmul(h, Rf) + Pf*c + bf)
            
            #cell state
            new_c = c * forget_gate + z * input_gate
            
            #output gate
            Wo = tf.get_variable("Wo", [input_size, self._num_units], dtype=dtype)
            Ro = tf.get_variable("Ro", [self._output_size, self._num_units], dtype=dtype)
            Po = tf.get_variable("Po", [batch_size, self._num_units], dtype=dtype)
            bo = tf.get_variable("bo", [1,  self._num_units], dtype=dtype)
            output_gate = tf.sigmoid(tf.matmul(inputs, Wo) + tf.matmul(h, Ro) + Po*c + bo)
            
            #block output
            new_h = output_gate * self._activation(new_c)
            if self._num_proj:
                w = tf.get_variable("w", [self._num_units, self._output_size], dtype=dtype)
                b = tf.get_variable("b", [1, self._output_size], dtype=dtype)
                new_h = tf.matmul(new_h, w) + b
            
            return new_h, (new_c, new_h)

class ConvLSTMCell(RNNCell):
    def __init__(self, num_units, shape, filter_size, 
                    num_proj=None, activation=tanh, reuse=None):
        '''
        num_units: the depth of filter
        shape: the size of input
        num_proj: the output channel
        '''
        self._num_units = num_units
        self._filter_size = filter_size
        self._shape = shape
        self._num_proj = num_proj
        self._activation = activation
        self._reuse = reuse

        if self._num_proj:
            self._output_size = [self._shape[0], self._shape[1],
                                    self._num_proj] 
        else:
            self._output_size = [self._shape[0], self._shape[1],
                                    self._num_units]


    @property
    def state_size(self):
        if self._num_proj is not None:
            tmp = (tf.TensorShape([self._shape[0], self._shape[1], 
                    self._num_units]), tf.TensorShape(self._output_size))
        else:
            tmp = (tf.TensorShape([self._shape[0], self._shape[1],
                     self._num_units]), tf.TensorShape([self._shape[0], 
                    self._shape[1], self._num_units])) 

        return LSTMStateTuple(tmp[0], tmp[1])
 

    @property
    def output_size(self):
        return tf.TensorShape(self._output_size) 

    '''
    def zero_state(self, batch_size, dtype):
        state_size = self.state_size
        return tuple([tf.zeros([batch_size] + s, dtype=dtype) for s in state_size])
    '''
    def __call__(self, inputs, state, scope=None):
        c, h = state
        dtype = inputs.dtype
        batch_size = inputs.get_shape().as_list()[0]
        in_h = inputs.get_shape().as_list()[1]
        in_w = inputs.get_shape().as_list()[2]
        channels = inputs.get_shape().as_list()[3] 
        f_h = self._filter_size[0]
        f_w = self._filter_size[1]

        with tf.variable_scope(scope or str(type(self).__name__), reuse=self._reuse):
            input_filter = [f_h, f_w, channels, self._num_units]
            recurrent_filter = [f_h, f_w, self._output_size[-1], self._num_units]
            peephole_weight = [1, in_h, in_w, self._num_units] 
            bias = [1, in_h, in_w, self._num_units]
            #input gate
            Wxi = tf.get_variable("Wxi", input_filter, dtype=dtype)
            Whi = tf.get_variable("Whi", recurrent_filter, dtype=dtype)
            Wci = tf.get_variable("Wci", peephole_weight, dtype=dtype)
            bi = tf.get_variable("bi", bias, dtype=dtype)
            input_gate = sigmoid(tf.nn.conv2d(inputs, Wxi, [1,1,1,1], "SAME") 
                                     + tf.nn.conv2d(h, Whi, [1,1,1,1], "SAME") + Wci*c + bi)

            #forget gate
            Wxf = tf.get_variable("Wxf", input_filter, dtype=dtype)
            Whf = tf.get_variable("Whf", recurrent_filter , dtype=dtype)
            Wcf = tf.get_variable("Wcf", peephole_weight, dtype=dtype)
            bf = tf.get_variable("bf", bias, dtype=dtype)
            forget_gate = sigmoid(tf.nn.conv2d(inputs, Wxf, [1,1,1,1], "SAME")
                                    + tf.nn.conv2d(h, Whf, [1,1,1,1], "SAME") + Wcf*c + bf)

            #new cell state
            Wxc = tf.get_variable("Wxc", input_filter, dtype=dtype)
            Whc = tf.get_variable("Whc", recurrent_filter, dtype=dtype)
            bc = tf.get_variable("bc", bias, dtype=dtype)
            new_c = forget_gate * c + input_gate * self._activation(tf.nn.conv2d(inputs, Wxc, [1,1,1,1], "SAME")
                             + tf.nn.conv2d(h, Whc, [1,1,1,1], "SAME") + bc)

            #output gate
            Wxo = tf.get_variable("Wxo", input_filter, dtype=dtype)
            Who = tf.get_variable("Who", recurrent_filter, dtype=dtype)
            Wco = tf.get_variable("Wco", peephole_weight, dtype=dtype)
            bo = tf.get_variable("bo", bias, dtype=dtype)
            output_gate = sigmoid(tf.nn.conv2d(inputs, Wxo, [1,1,1,1], "SAME")
                                    + tf.nn.conv2d(h, Who, [1,1,1,1], "SAME") + Wco*c + bo)

            #new h
            new_h = output_gate * self._activation(new_c)
            if self._num_proj:
                w = tf.get_variable("w", [in_h, in_w, self._num_units, self._num_proj])
                new_h = tf.nn.conv2d(new_h, w, [1, 1, 1, 1], "SAME") 

            if self._reuse == None:
                self._reuse = True

            return new_h, LSTMStateTuple(new_c, new_h)
