import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
import tensorflow as tf
import sys
from custom_cell import LSTMCell

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
            self.enc_cell = tf.contrib.rnn.LSTMCell(self.hidden_num, use_peepholes=True)
            self.dec_cell = tf.contrib.rnn.LSTMCell(self.hidden_num, use_peepholes=True)
        else:
            self.enc_cell = enc_cell
            self.dec_cell = dec_cell

        with tf.variable_scope('encode', reuse=None):
            _, enc_state = tf.nn.dynamic_rnn(self.enc_cell, inputs, dtype=tf.float32, time_major=True)

        saved_output = tf.Variable(tf.zeros([batch_size, self.hidden_num]), trainable=False)
        saved_state = tf.Variable(tf.zeros([batch_size, self.hidden_num]), trainable=False)

        #projection weights and biases
        w = tf.Variable(tf.truncated_normal([self.hidden_num, desired], -0.1 , 0.1, dtype=tf.float32), name='w')
        b = tf.Variable(tf.zeros([desired]), name='b', dtype=tf.float32)

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
            outputs = [tf.matmul(i, w) + b for i in outputs]
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
        #gradients, _ = tf.clip_by_global_norm(gradients, 5)
        gradients = [tf.clip_by_value(grad, -1, 1) for grad in gradients]
        self.train = self.optimizer.apply_gradients(
                    zip(gradients, v), global_step=global_step)
        
        self.loss_sum = tf.summary.scalar('loss', self.loss)

if __name__ =='__main__':
    PATH = "/home/stud/wangc/lab/record/"
    DATASET = "../mnist.h5"
    f = open(PATH + "log", "w+")  
    maxtime = 20
    desired = 64 * 64
    hidden_num = 2500
    batch_size = 50
    data_generator = DataHandler(DATASET, num_frames=maxtime, batch_size=batch_size)

    epoch = 300
    steps = 200
    val_steps = 20
    test_steps = 20

    enc_cell = LSTMCell(2000)
    dec_cell = LSTMCell(2000)

    inputs = tf.placeholder(tf.float32, shape = [maxtime, batch_size, desired], name='inputs')
    rmsOpti = tf.train.RMSPropOptimizer(0.001)

    ae = Autoencoder(inputs, hidden_num, optimizer=rmsOpti, enc_cell=enc_cell, dec_cell=dec_cell) 
    print("hidden_num %d, batch_size %d, epoch %d, optimizer %s, cell %s" % (hidden_num, batch_size, epoch, ae.optimizer, ae.enc_cell), file=f)
    f.flush()

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        train_writer = tf.summary.FileWriter(PATH + 'logdir'+'/train', sess.graph)
        validate_writer = tf.summary.FileWriter(PATH + 'logdir'+'/validate', sess.graph) 

        for j in range(epoch):
            for i in range(steps):
                _, train_sum = sess.run([ae.train, ae.loss_sum],
                                        feed_dict={inputs:data_generator.get_batch().reshape(maxtime, batch_size, -1)})
            train_writer.add_summary(train_sum, j)
            
            val_loss_sum = 0
            for p in range(val_steps): 
                val_sum, val_loss = sess.run([ae.loss_sum, ae.loss],
                                             feed_dict={inputs:data_generator.get_batch().reshape(maxtime, batch_size, -1)})
                val_loss_sum += val_loss
            val_avrg = val_loss_sum/val_steps
            val_summa = tf.Summary(value=[
                tf.Summary.Value(tag="loss", simple_value=val_avrg),])
            validate_writer.add_summary(val_summa, j)
            

        test_sum = 0
        for k in range(test_steps):
            test_ins = data_generator.get_batch().reshape(maxtime, batch_size, -1)
            test_outputs, test_l = sess.run([ae.outputs, ae.loss], 
                                            feed_dict={inputs:test_ins})
            test_sum += test_l 
        average_test = test_sum/test_steps
        print("test error %f" % average_test, file=f)

        np.savez_compressed(PATH + "outputs",
                test_out=test_outputs, test_in=test_ins)

    f.close()
    sys.exit(0)
