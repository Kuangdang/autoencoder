import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
import tensorflow as tf
import sys
from lstmcell import LSTMCell

#autoencoder class
class Autoencoder:
    def __init__(self, inputs, hidden_num, enc_cell=None, dec_cell=None, optimizer=None):
        '''
        inputs shape [maxtime, batch_size, frame_size*frame_size]
        '''
        maxtime = inputs.get_shape().as_list()[0]
        batch_size = inputs.get_shape().as_list()[1]
        desired = inputs.get_shape().as_list()[2]
        
        self.hidden_num = hidden_num
        if enc_cell is None:
            self.enc_cell = tf.contrib.rnn.LSTMCell(self.hidden_num, use_peepholes=True, num_proj=desired)
            self.dec_cell = tf.contrib.rnn.LSTMCell(self.hidden_num, use_peepholes=True, num_proj=desired)
        else:
            self.enc_cell = enc_cell
            self.dec_cell = dec_cell

        with tf.variable_scope('encode', reuse=None):
            _, enc_state = tf.nn.dynamic_rnn(self.enc_cell, inputs, dtype=tf.float32, time_major=True)

        saved_output = tf.Variable(tf.zeros([batch_size, desired]), trainable=False)
        saved_state = tf.Variable(tf.zeros([batch_size, desired]), trainable=False)

        outputs = list()
        output = saved_output
        state = saved_state

        with tf.variable_scope('dec_scope', reuse=None):
            dec_inputs = tf.zeros([batch_size, desired])
            dec_state = enc_state
            v = None
            for _ in range(maxtime):
                with tf.variable_scope('unrolling', reuse=v):
                    output, dec_state = self.dec_cell(dec_inputs, dec_state)
                    v = True
                    outputs.append(output)

        #state saving across unrolling
        with tf.control_dependencies([saved_output.assign(output),
                                      saved_state.assign(state)]):
            self.outputs = tf.stack(outputs)
            #reverse the outputs to make the training easier
            reversed_outputs = outputs[::-1]
            self.loss = tf.reduce_mean(tf.squared_difference(reversed_outputs, inputs))

        #optimizer
        if optimizer is None:
            learning_rate_start = 0.001
            self.optimizer = tf.train.AdamOptimizer(learning_rate_start)
        else:
            self.optimizer = optimizer 
        global_step = tf.Variable(0, trainable = False)
        gradients, v = zip(*self.optimizer.compute_gradients(self.loss))
        gradients, _ = tf.clip_by_global_norm(gradients, 5)
        self.train = self.optimizer.apply_gradients(
                    zip(gradients, v), global_step=global_step)
        
        self.loss_sum = tf.summary.scalar('loss', self.loss)

if __name__ =='__main__':
    PATH = "/home/stud/wangc/lab/record/"
    f = open(PATH + "log", "w+")  
    data = np.load('/home/stud/wangc/lab/mnist_test_seq.npy')
    data = np.around(data/255, decimals=5)
    data = data.reshape(data.shape[0],data.shape[1],-1)
    maxtime = data.shape[0]
    desired = data.shape[2]
    hidden_num = 2000
    batch_size = 50
    train_size = 9000
    test_size = 500
    epoch = 80
    steps = int(train_size/batch_size)

    inputs = tf.placeholder(tf.float32, shape = [maxtime, batch_size, desired], name='inputs')
    ae = Autoencoder(inputs, hidden_num)
    print("hidden_num %d, batch_size %d, epoch %d, optimizer %s, cell %s" % (hidden_num, batch_size, epoch, ae.optimizer, ae.enc_cell), file=f)
    f.flush()
    print('create session')

    with tf.Session() as sess:
        print('beginning-------------------------------')
        tf.global_variables_initializer().run()
        print('initialized')
        train_writer = tf.summary.FileWriter(
                PATH + 'logdir'+'/train', sess.graph)
        validate_writer = tf.summary.FileWriter(
                PATH + 'logdir'+'/validate', sess.graph) 
        val_outputs = list()
        for j in range(epoch):
            for i in range(steps):
                _, train_sum = sess.run([ae.train, ae.loss_sum], feed_dict={inputs:(data[:,i*batch_size:(i+1)*batch_size])})
            train_writer.add_summary(train_sum, j)
            _, val_sum = sess.run([ae.outputs, ae.loss_sum], feed_dict={inputs:(data[:,train_size:train_size+batch_size])})
            validate_writer.add_summary(val_sum, j)
        val_outputs = np.array(val_outputs)
        train_writer.close()
        validate_writer.close()

        test_sum = 0
        test_steps = int(test_size/batch_size))
        for k in range(test_steps):
            test_outputs, test_l = sess.run([ae.outputs, ae.loss], feed_dict={inputs:data[:,9500+k*batch_size:9500+(k+1)*batch_size]})
            test_sum += test_l 
        print("test error %f" % test_sum/test_steps, file=f)

        np.savez_compressed(PATH + "outputs",
                test_out=test_outputs, test_in=data[:,-batch_size:])
    
    print('end---------------------------------')
    f.close()
    sys.exit(0)