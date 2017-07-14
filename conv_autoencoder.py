import numpy as np
import tensorflow as tf
import sys
from custom_cell import ConvLSTMCell

#autoencoder class
class Autoencoder:
    def __init__(self, inputs, enc_cell, dec_cell, optimizer=None):
        '''
        inputs shape [maxtime, batch_size, height, weight, channels]
        '''
        maxtime = inputs.get_shape().as_list()[0]
        batch_size = inputs.get_shape().as_list()[1]
        in_h  = inputs.get_shape().as_list()[2]
        in_w  = inputs.get_shape().as_list()[3]
        channels  = inputs.get_shape().as_list()[4]

        
        self.hidden_num = enc_cell._num_units
        self.enc_cell = enc_cell
        self.dec_cell = dec_cell

        with tf.variable_scope('encode', reuse=None):
            _, enc_state = tf.nn.dynamic_rnn(self.enc_cell, inputs, dtype=tf.float32, time_major=True)

        saved_output = tf.Variable(tf.zeros([batch_size, in_h, in_w, self.dec_cell.output_size[-1]]), trainable=False)
        saved_state = tf.Variable(tf.zeros([batch_size, in_h, in_w, self.hidden_num]), trainable=False)


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
    DATASET = "/home/stud/wangc/lab/mnist_test_seq.npy"
    f = open(PATH + "log", "w+") 
    data = np.load(DATASET)
    data = np.around(data/255, decimals=5)
    data = data.reshape(data.shape[0], data.shape[1], data.shape[2], data.shape[3], 1)
    maxtime = data.shape[0]
    in_h = data.shape[2]
    in_w = data.shape[3]
    batch_size = 50
    train_size = 9000
    val_size = 500
    test_size = 500
    epoch = 50
    steps = int(train_size/batch_size)
    val_steps = int(val_size/batch_size)
    test_steps = int(test_size/batch_size)
    val_start = train_size
    test_start = val_start + val_size

    enc_cell = ConvLSTMCell(30, (in_h, in_w), [6,6], 1)
    dec_cell = ConvLSTMCell(30, (in_h, in_w), [6,6], 1)

    inputs = tf.placeholder(tf.float32, shape = [maxtime, batch_size, in_h, in_w, 1], name='inputs')
    ae = Autoencoder(inputs, enc_cell=enc_cell, dec_cell=dec_cell) 
    print("hidden_num %d, batch_size %d, epoch %d, optimizer %s, cell %s" 
            % (ae.enc_cell._num_units, batch_size, epoch, ae.optimizer, ae.enc_cell), file=f)
    f.flush()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        train_writer = tf.summary.FileWriter(
                PATH + 'logdir'+'/train', sess.graph)
        validate_writer = tf.summary.FileWriter(
                PATH + 'logdir'+'/validate', sess.graph) 
        for j in range(epoch):
            for i in range(steps):
                _, train_sum = sess.run([ae.train, ae.loss_sum], feed_dict={inputs:(data[:,i*batch_size:(i+1)*batch_size])})
            train_writer.add_summary(train_sum, j)

            val_loss_sum = 0
            for p in range(val_steps): 
                _, val_sum, val_loss = sess.run([ae.outputs, ae.loss_sum, ae.loss],
                        feed_dict={inputs:(data[:,val_start+p*batch_size:val_start+(p+1)*batch_size])})
                val_loss_sum += val_loss
            val_avrg = val_loss_sum/val_steps
            val_summa = tf.Summary(value=[
                tf.Summary.Value(tag="loss", simple_value=val_avrg),])
            validate_writer.add_summary(val_summa, j)

        test_sum = 0
        for k in range(test_steps):
            test_outputs, test_l = sess.run([ae.outputs, ae.loss], 
                    feed_dict={inputs:data[:,test_start+k*batch_size:test_start+(k+1)*batch_size]})
            test_sum += test_l 
        average_test = test_sum/test_steps
        print("test error %f" % average_test, file=f)

        np.savez_compressed(PATH + "outputs",
                test_out=test_outputs, test_in=data[:,-batch_size:-1])
     
    f.close()
    sys.exit(0)
