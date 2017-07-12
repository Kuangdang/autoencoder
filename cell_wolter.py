from tensorflow.python.ops.rnn_cell_impl import RNNCell                                                                        
from tensorflow.python.ops.rnn_cell_impl import LSTMStateTuple
import tensorflow as tf
import numpy as np
#from tensorflow.contrib.rnn.python.ops.core_rnn_cell_impl import _checked_scope
from tensorflow.python.ops.math_ops import tanh
from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops import variable_scope as vs
class ClassicConvLSTM(RNNCell):
    def __init__(self, kernel_size, depth, input_dims, output_depth=None,
            strides=None, reuse=None):
        self.reuse = reuse
        self.kernel_size = kernel_size
        self.depth = depth
        self.input_dims = [int(dim) for dim in input_dims]
        if strides is None:
            self.strides = [1, 1]
        else:
            self.strides = strides
            self.output_depth = output_depth

    @property
    def recurrent_size(self):
        return tf.TensorShape([np.ceil(self.input_dims[0] / self.strides[0]),
            np.ceil(self.input_dims[1] / self.strides[1]), self.depth])

    @property
    def output_size(self):
        if self.output_depth is None:
            return self.recurrent_size
        else:
            total_els = (np.ceil(self.input_dims[0] / self.strides[0])
                * np.ceil(self.input_dims[1] / self.strides[1])
                * self.output_depth)
            out_depth = total_els / (self.input_dims[0]*self.input_dims[1])
            return tf.TensorShape([self.input_dims[0],
                self.input_dims[1], out_depth])

    @property
    def state_size(self):
        return LSTMStateTuple(self.recurrent_size, self.recurrent_size)

    def __call__(self, inputs, state, scope=None):
        input_shape = tf.Tensor.get_shape(inputs)
        num_channels = int(input_shape[-1])

        c, h = state
        with tf.variable_scope(scope or str(type(self).__name__), reuse=self.reuse):
            with tf.variable_scope('forward_gate_weights'):
                variable_shape = self.kernel_size + [num_channels] + [self.depth]
                Wi = tf.get_variable('input_weights', variable_shape)
                Wf = tf.get_variable('forget_weights', variable_shape)
                Wo = tf.get_variable('output_weights', variable_shape)
                Wz = tf.get_variable('state_weights', variable_shape)

            with tf.variable_scope('recurrent_gate_weights'):
                variable_shape = self.kernel_size + [self.depth]*2
                Ri = tf.get_variable('recurrent_input_weights', variable_shape)
                Rf = tf.get_variable('recurrent_forget_weights', variable_shape)
                Ro = tf.get_variable('recurrent_output_weights', variable_shape)
                Rz = tf.get_variable('recurrent_state_weights', variable_shape)

            with tf.variable_scope('peephole_weights'):
                pi = tf.get_variable('input_peep', self.state_size[0])
                pf = tf.get_variable('forget_peep', self.state_size[0])
                po = tf.get_variable('output_peep', self.state_size[0])

            with tf.variable_scope('biases'):
                bi = tf.get_variable('input_bias', self.state_size[0])
                bf = tf.get_variable('forget_bias', self.state_size[0])
                bo = tf.get_variable('output_bias', self.state_size[0])
                bz = tf.get_variable('state_bias', self.state_size[0])

            ii_conv = tf.nn.convolution(inputs, Wi, strides=self.strides, padding='SAME', name='input_conv')
            ri_conv = tf.nn.convolution(h, Ri, padding='SAME', name='recurrent_input_conv')
            i = tf.nn.sigmoid(ii_conv + ri_conv + c*pi + bi, name='input_gate')
            if_conv = tf.nn.convolution(inputs, Wf, strides=self.strides, padding='SAME', name='forget_conv')
            rf_conv = tf.nn.convolution(h, Rf, padding='SAME', name='recurrent_forget_conv')
            f = tf.nn.sigmoid(if_conv + rf_conv + c*pf + bf, name='forget_gate')

            iz_conv = tf.nn.convolution(inputs, Wz, padding='SAME',
                strides=self.strides, name='condidate_conv')
            rz_conv = tf.nn.convolution(h, Rz, padding='SAME',
                name='recurrent_condidate_conv')
            z = tf.nn.tanh(iz_conv + rz_conv + bz, name='candidates')
            c = z*i + c*f
            io_conv = tf.nn.convolution(inputs, Wo, strides=self.strides,
                padding='SAME', name='output_conv')
            ro_conv = tf.nn.convolution(h, Ro, padding='SAME',
                name='recurrent_output_conv')
            o = tf.nn.sigmoid(io_conv + ro_conv + c*po + bo, name='output_gate')
            h = tf.nn.tanh(c)*o
            if self.output_depth is not None:
                variable_shape = self.kernel_size + [self.depth] + [self.output_depth]
                Wproj = tf.get_variable('projection_weights', variable_shape)
                output = tf.nn.convolution(h, Wproj, padding='SAME')
                output = tf.reshape(output, input_shape)
            else:
                output = h

            new_state_touple = LSTMStateTuple(c, h)

            if self.reuse is None:
                self.reuse = True

            return output, new_state_touple
