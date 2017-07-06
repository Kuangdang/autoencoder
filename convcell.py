from tensorflow.python.ops.rnn_cell_impl import _RNNCell as RNNCell                                                                        
import tensorflow as tf
#from tensorflow.contrib.rnn.python.ops.core_rnn_cell_impl import _checked_scope
from tensorflow.python.ops.math_ops import tanh
from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops import variable_scope as vs


class ConvLSTMCell(RNNCell):
    def __init__(self, num_units, shape, filter_size, activation=tanh, reuse=None):
        self._num_units = num_units
        self._filter_size = filter_size
        self._shape = shape
        self._activation = activation
        self._reuse = reuse


    @property
    def state_size(self):
        return ([self._shape[0], self._shape[1], self._num_units], [self._shape[0], self._shape[1], self._num_units])

    @property
    def output_size(self):
        return [self._shape[0], self._shape[1], self._num_units]

    def zero_state(self, batch_size, dtype):
        state_size = self.state_size
            return tuple([tf.zeros([batch_size] + s], dtype=dtype) for s in state_size])

    def __call__(self, inputs, state, scope=None):
        c, h = state
        dtype = inputs.dtype
        batch_size = inputs.get_shape()[0]
        in_h = inputs.get_shape()[1]
        in_w = inputs.get_shape()[2]
        channels = inputs.get_shape()[3] 
        f_h = self._filter_size[0]
        f_w = self._filter_size[1]
        #input gate
        Wxi = tf.get_variable("Wxi", [f_h, f_w, channels, self._num_units], dtype=dtype)
        Whi = tf.get_variable("Whi", [f_h, f_w, channels, self._num_units], dtype=dtype)
        Wci = tf.get_variable("Wci", [1, in_h, in_w, self._num_units], dtype=dtype)
        bi = tf.get_variable("bi", [1, in_h, in_w, self._num_units], dtype=dtype)
        input_gate = sigmoid(tf.nn.conv2d(inputs, Wxi, [1,1,1,1], "SAME") + tf.nn.conv2d(h, Whi, [1,1,1,1], "SAME")
                + Wci*c + bi)

        #forget gate
        Wxf = tf.get_variable("Wxf", [f_h, f_w, channels, self._num_units], dtype=dtype)
        Whf = tf.get_variable("Whf", [f_h, f_w, channels, self._num_units], dtype=dtype)
        Wcf = tf.get_variable("Wcf", [1, in_h, in_w, self._num_units], dtype=dtype)
        bf = tf.get_variable("bf", [1, in_h, in_w, self._num_units], dtype=dtype)
        forget_gate = sigmoid(tf.nn.conv2d(inputs, Wxf, [1,1,1,1], "SAME") + tf.nn.conv2d(h, Whf, [1,1,1,1], "SAME")
                + Wcf*c + bf)

        #new cell state
        Wxc = tf.get_variable("Wxc", [f_h, f_w, channels, self._num_units], dtype=dtype)
        Whc = tf.get_variable("Whc", [f_h, f_w, channels, self._num_units], dtype=dtype)
        bc = tf.get_variable("bc", [1, in_h, in_w, self._num_units], dtype=dtype)
        new_c = forget_gate * c + input_gate * self._activation(tf.nn.conv2d(inputs, Wxc, [1,1,1,1], "SAME")
                + tf.nn.conv2d(h, Whc, [1,1,1,1]), "SAME" + bc)

        #output gate
        Wxo = tf.get_variable("Wxo", [f_h, f_w, channels, self._num_units], dtype=dtype)
        Who = tf.get_variable("Who", [f_h, f_w, channels, self._num_units], dtype=dtype)
        Wco = tf.get_variable("Wco", [1, in_h, in_w, self._num_units], dtype=dtype)
        bo = tf.get_variable("bo", [1, in_h, in_w, self._num_units], dtype=dtype)
        output_gate = sigmoid(tf.nn.conv2d(inputs, Wxo, [1,1,1,1], "SAME") + tf.nn.conv2d(h, Who, [1,1,1,1], "SAME")
                + Wco*c + bo)

        #new h
        new_h = output_gate * self._activation(new_c)

        return new_h, (new_c, new_h)
