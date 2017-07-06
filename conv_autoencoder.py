import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
import tensorflow as tf
import sys
from convcell import ConvLSTMCell

#autoencoder class
class Autoencoder:
    def __init__(self, inputs, hidden_num, enc_cell, dec_cell, optimizer=None):
        '''
        inputs shape [maxtime, batch_size, height, weight, channels]
        '''
        maxtime = inputs.get_shape().as_list()[0]
        batch_size = inputs.get_shape().as_list()[1]
        in_h  = inputs.get_shape().as_list()[2]
        in_w  = inputs.get_shape().as_list()[3]
        channels  = inputs.get_shape().as_list()[4]

        
        self.hidden_num = hidden_num
        self.enc_cell = enc_cell
        self.dec_cell = dec_cell

        with tf.variable_scope('encode', reuse=None):
            _, enc_state = tf.nn.dynamic_rnn(self.enc_cell, inputs, dtype=tf.float32, time_major=True)

        saved_output = tf.Variable(tf.zeros([batch_size, in_h, in_w, self.hidden_num]), trainable=False)
        saved_state = tf.Variable(tf.zeros([batch_size, in_h, in_w, self.hidden_num]), trainable=False)

        #projection weights and biases
        w = tf.Variable(tf.truncated_normal([1, 1, self.hidden_num, channels], -0.1 , 0.1, dtype=tf.float32), name='w')
        b = tf.Variable(tf.zeros([1, in_h, in_w, channels]), name='b', dtype=tf.float32)

        outputs = list()
        output = saved_output
        state = saved_state

        with tf.variable_scope('dec_scope', reuse=None):
            dec_inputs = tf.zeros([batch_size, in_h, in_w, channels])
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
            outputs = [tf.nn.conv2d(i, w, [1, 1, 1, 1], "SAME") + b for i in outputs]
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
    f = open("/home/stud/wangc/lab/record/log", "w+")  
    data = np.load('/home/stud/wangc/lab/mnist_test_seq.npy')
    data = np.around(data/255, decimals=5)
    data = data.reshape(data.shape[0],data.shape[1], data.shape[2], data.shape[3], 1)
    maxtime = data.shape[0]
    in_h = data.shape[2]
    in_w = data.shape[3]
    hidden_num = 1000
    batch_size = 50
    train_size = 9000
    epoch = 200
    steps = int(train_size/batch_size)

    enc_cell = ConvLSTMCell(1000, (in_h, in_w), [6,6])
    dec_cell = ConvLSTMCell(1000, (in_h, in_w), [6,6])

    inputs = tf.placeholder(tf.float32, shape = [maxtime, batch_size, in_h, in_w, 1], name='inputs')
    ae = Autoencoder(inputs, hidden_num, enc_cell=enc_cell, dec_cell=dec_cell) 
    print("hidden_num %d, batch_size %d, epoch %d, optimizer %s, cell %s" % (hidden_num, batch_size, epoch, ae.optimizer, ae.enc_cell), file=f)
    f.flush()
    with tf.Session() as sess:
        print('beginning-------------------------------')
        tf.global_variables_initializer().run()
        print('initialized')
        train_writer = tf.summary.FileWriter(
                '/home/stud/wangc/lab/record/logdir'+'/train', sess.graph)
        validate_writer = tf.summary.FileWriter(
                '/home/stud/wangc/lab/record/logdir'+'/validate', sess.graph) 
        val_outputs = list()
        for j in range(epoch):
            for i in range(steps):
                _, train_sum = sess.run([ae.train, ae.loss_sum], feed_dict={inputs:(data[:,i*batch_size:(i+1)*batch_size])})
            train_writer.add_summary(train_sum, j)
            val_output, val_sum = sess.run([ae.outputs, ae.loss_sum], feed_dict={inputs:(data[:,train_size:train_size+batch_size])})
            validate_writer.add_summary(val_sum, j)
            if j%20 == 0:
                val_outputs.append(val_output)
        val_outputs = np.array(val_outputs)
        train_writer.close()
        validate_writer.close()

        test_outputs, test_l = sess.run([ae.outputs, ae.loss], feed_dict={inputs:data[:,9500:9500+batch_size]})
        print("test error %f" % test_l, file=f)

        np.savez_compressed("/home/stud/wangc/lab/record/outputs",
                val_out=val_outputs, val_in=data[:,9000:9000+batch_size], test_out=test_outputs, test_in=data[:,9500:9500+batch_size])
    
    print('end---------------------------------')
    f.close()
    sys.exit(0)
