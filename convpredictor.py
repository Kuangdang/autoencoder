import sys
import numpy as np
import tensorflow as tf
from custom_cell import ConvLSTMCell
from new_handler import DataHandler

#autoencoder class
class ConvPredictor:
    def __init__(self, inputs, predict_frames_num, enc_cell, dec_cell, targets=None, optimizer=None, conditioned=None):
        '''
        inputs shape [maxtime, batch_size, height, weight, channels]
        '''
        maxtime = inputs.get_shape().as_list()[0]
        batch_size = inputs.get_shape().as_list()[1]
        in_h = inputs.get_shape().as_list()[2]
        in_w = inputs.get_shape().as_list()[3]
        channels = inputs.get_shape().as_list()[4]
        self.conditioned = conditioned

        
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
                    if conditioned is None:
                        output, dec_state = self.dec_cell(dec_inputs, dec_state)
                    else:
                        if v is None:
                            output, dec_state = self.dec_cell(dec_inputs, dec_state)
                        else:
                            output, dec_state = self.dec_cell(output, dec_state)
                        
                    v = True
                    outputs.append(output)


        #state saving across unrolling
        with tf.control_dependencies([saved_output.assign(output),
                                      saved_state.assign(state)]):
            self.outputs = tf.stack(outputs)
            #reverse the outputs to make the training easier
            self.loss = tf.reduce_mean(tf.squared_difference(self.outputs, targets))

        #optimizer
        if optimizer is None:
            learning_rate_start = 0.001
            self.optimizer = tf.train.AdamOptimizer(learning_rate_start)
        else:
            self.optimizer = optimizer 
        global_step = tf.Variable(0, trainable = False)
        gradients, v = zip(*self.optimizer.compute_gradients(self.loss))
        gradients = [tf.clip_by_value(grad, -1, 1) for grad in gradients]
        self.train = self.optimizer.apply_gradients(
                    zip(gradients, v), global_step=global_step)
        
        self.loss_sum = tf.summary.scalar('loss', self.loss)

if __name__ == '__main__':
    PATH = "/home/wangc/lab/record/"
    DATASET = "../mnist.h5"
    save_path = "/home/wangc/lab/record/model.ckpt"
    f = open(PATH + "log", "w+") 
    input_frames = 10
    predict_frames = 10
    total_frames = input_frames + predict_frames
    in_h = 64
    in_w = 64
    batch_size = 30
    data_generator = DataHandler(DATASET, num_frames=total_frames, batch_size=batch_size)

    epoch = 200
    steps = 300
    val_steps = 50
    test_steps = 50

    enc_cell = ConvLSTMCell(30, (in_h, in_w), [8,8], 1)
    dec_cell = ConvLSTMCell(30, (in_h, in_w), [8,8], 1)

    inputs = tf.placeholder(tf.float32, shape = [input_frames, batch_size, in_h, in_w, 1], name='inputs')
    targets = tf.placeholder(tf.float32, shape = [predict_frames, batch_size, in_h, in_w, 1], name='targets')

    rmsOpti = tf.train.RMSPropOptimizer(0.001)
    ae = ConvPredictor(inputs, predict_frames=10, enc_cell=enc_cell, dec_cell=dec_cell, targets=targets,
                       optimizer=rmsOpti, conditioned=False) 
    print("class %s, features %d, batch_size %d, epoch %d, optimizer %s, cell %s, conditioned %s" 
            % (type(ae).__name__, ae.enc_cell._num_units, batch_size, epoch, ae.optimizer, ae.enc_cell, ae.conditioned), file=f)
    f.flush()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        train_writer = tf.summary.FileWriter(
                PATH + 'logdir'+'/train', sess.graph)
        validate_writer = tf.summary.FileWriter(
                PATH + 'logdir'+'/validate', sess.graph) 
        for j in range(epoch):
            for i in range(steps):
                data = data_generator.get_batch().reshape(maxtime, batch_size, in_h, in_w, 1)
                _, train_sum = sess.run([ae.train, ae.loss_sum],
                                        feed_dict={inputs:data[:10], targets:data[10:]}
                train_writer.add_summary(train_sum, j)

            val_loss_sum = 0
            for p in range(val_steps): 
                data = data_generator.get_batch().reshape(maxtime, batch_size, in_h, in_w, 1)
                val_sum, val_loss = sess.run([ae.loss_sum, ae.loss],
                                             feed_dict={inputs:data[:10], targets:data[10:]}
                val_loss_sum += val_loss
            val_avrg = val_loss_sum/val_steps
            val_summa = tf.Summary(value=[tf.Summary.Value(tag="loss", simple_value=val_avrg),])
            validate_writer.add_summary(val_summa, j)
        saver.save(sess, save_path)
        
        '''
        test_sum = 0
        for k in range(test_steps):
            test_ins = data_generator.get_batch().reshape(maxtime, batch_size, in_h, in_w, 1)
            test_outputs, test_l = sess.run([ae.outputs, ae.loss], 
                                        feed_dict={inputs:test_ins})
            test_sum += test_l 
        average_test = test_sum/test_steps
        print("test error %f" % average_test, file=f)

        np.savez_compressed(PATH + "outputs",
                test_out=test_outputs, test_in=test_ins)
        '''
     
    f.close()
    sys.exit(0)
