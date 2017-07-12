from tensorflow.python.ops.rnn_cell_impl import _RNNCell as RNNCell
import tensorflow as tf
#from tensorflow.contrib.rnn.python.ops.core_rnn_cell_impl import _checked_scope
from tensorflow.python.ops.math_ops import tanh
from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops import variable_scope as vs

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
            Wz = tf.get_variable("Wz", [input_size, self._num_units], dtype=dtype)
            Rz = tf.get_variable("Rz", [self._output_size, self._num_units], dtype=dtype)
            bz = tf.get_variable("bz", [1,  self._num_units], dtype=dtype)
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
