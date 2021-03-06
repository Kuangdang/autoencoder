25928import sys
import numpy as np
import tensorflow as tf
from new_handler import DataHandler
#autoencoder class
class CompositeModel:
    def __init__(self, inputs, predict_frames, hidden_num, targets=None, enc_cell=None, dec_cell=None, optimizer=None, conditioned=None):
        '''
        inputs shape [input_frames, batch_size, frame_size*frame_size]
        '''
        input_frames = inputs.get_shape().as_list()[0]
        batch_size = inputs.get_shape().as_list()[1]
        desired = inputs.get_shape().as_list()[2]
        self.conditioned = conditioned

        self.hidden_num = hidden_num
        if enc_cell is None:
            self.enc_cell = tf.contrib.rnn.LSTMCell(self.hidden_num,
                                                    use_peepholes=True, num_proj=desired)
            self.dec_reconstruct_cell = tf.contrib.rnn.LSTMCell(self.hidden_num,
                                                    use_peepholes=True, num_proj=desired)
            self.dec_predict_cell = tf.contrib.rnn.LSTMCell(self.hidden_num,
                                                    use_peepholes=True, num_proj=desired)
        else:
            self.enc_cell = enc_cell
            self.dec_reconstruct_cell = dec_cell
            self.dec_predict_cell = dec_cell

        with tf.variable_scope('encode', reuse=None):
            _, enc_state = tf.nn.dynamic_rnn(self.enc_cell,
                                             inputs, dtype=tf.float32, time_major=True)

        #assum projection is performed in cell
        saved_reconstruct_output = tf.Variable(tf.zeros([batch_size, desired]), trainable=False)
        saved_reconstruct_state = tf.Variable(tf.zeros([batch_size, desired]), trainable=False)

        saved_predict_output = tf.Variable(tf.zeros([batch_size, desired]), trainable=False)
        saved_predict_state = tf.Variable(tf.zeros([batch_size, desired]), trainable=False)

        reconstruct_outputs = list()
        reconstruct_output = saved_reconstruct_output
        reconstruct_state = saved_reconstruct_state


        predict_outputs = list()
        predict_output = saved_predict_output
        predict_state = saved_predict_state

        with tf.variable_scope('predict_scope', reuse=None):
            dec_predict_inputs = tf.zeros([batch_size, desired])
            dec_predict_state = enc_state
            v = None
            for _ in range(predict_frames):
                with tf.variable_scope('unrolling', reuse=v):
                    if conditioned is None:
                        predict_output, dec_predict_state = self.dec_predict_cell(dec_predict_inputs, dec_predict_state)
                    else:
                        if v is None:
                            predict_output, dec_predict_state = self.dec_predict_cell(dec_predict_inputs, dec_predict_state)
                        else:
                            predict_output, dec_predict_state = self.dec_predict_cell(predict_output, dec_predict_state)

                    v = True
                    predict_outputs.append(predict_output)

        with tf.variable_scope('reconstruct_scope', reuse=None):
            dec_reconstruct_inputs = tf.zeros([batch_size, desired])
            dec_reconstruct_state = enc_state
            v = None
            for _ in range(input_frames):
                with tf.variable_scope('unrolling', reuse=v):
                    '''
                    if conditioned is None:
                        reconstruct_output, dec_reconstruct_state = self.dec_reconstruct_cell(dec_reconstruct_inputs,
                                                                                              dec_reconstruct_state)
                    else:
                        if v is None:
                            reconstruct_output, dec_reconstruct_state = self.dec_reconstruct_cell(dec_reconstruct_inputs,
                                                                                                  dec_reconstruct_state)
                        else:
                            reconstruct_output, dec_reconstruct_state = self.dec_reconstruct_cell(reconstruct_output,
                                                                                                  dec_reconstruct_state)
                    '''
                    reconstruct_output, dec_reconstruct_state = self.dec_reconstruct_cell(dec_reconstruct_inputs,
                                                                                              dec_reconstruct_state)

                    v = True
                    reconstruct_outputs.append(reconstruct_output)

        #state saving across unrolling
        with tf.control_dependencies([saved_predict_output.assign(predict_output),
                                      saved_predict_state.assign(predict_state)]):
            self.predict_outputs = tf.stack(predict_outputs)

        with tf.control_dependencies([saved_reconstruct_output.assign(reconstruct_output),
                                      saved_reconstruct_state.assign(reconstruct_state)]):
            self.reconstruct_outputs = tf.stack(reconstruct_outputs)

        self.predict_loss = tf.reduce_mean(tf.squared_difference(self.predict_outputs, targets))
        self.reconstruct_loss = tf.reduce_mean(tf.squared_difference(self.reconstruct_outputs[::-1], inputs))
        #more weight on predict loss
        self.loss = 0.7*self.predict_loss + 0.3*self.reconstruct_loss

        #optimizer
        if optimizer is None:
            learning_rate_start = 0.001
            self.optimizer = tf.train.AdamOptimizer(learning_rate_start)
        else:
            self.optimizer = optimizer
        global_step = tf.Variable(0, trainable=False)
        gradients, v = zip(*self.optimizer.compute_gradients(self.loss))
        #gradients, _ = tf.clip_by_global_norm(gradients, 5)
        #clip by value instead of the norm of all gradients
        gradients = [tf.clip_by_value(grad, -1, 1) for grad in gradients]
        self.train = self.optimizer.apply_gradients(
            zip(gradients, v), global_step=global_step)

        self.loss_sum = tf.summary.scalar('loss', self.loss)

if __name__ == '__main__':
    PATH = "/home/wangc/lab/record/"
    DATASET = "../mnist.h5"
    f = open(PATH + "log", "w+")
    save_path = "/home/wangc/lab/record/model.ckpt"
    input_frames = 10
    predict_frames = 10
    total_frames = input_frames + predict_frames
    desired = 64 * 64
    hidden_num = 1500
    batch_size = 30
    data_generator = DataHandler(DATASET, num_frames=total_frames, batch_size=batch_size)

    epoch = 500
    steps = 500
    val_steps = 50
    test_steps = 50

    inputs = tf.placeholder(tf.float32, shape=[input_frames, batch_size, desired], name='inputs')
    targets = tf.placeholder(tf.float32, shape=[predict_frames, batch_size, desired], name='targets')

    rmsOpti = tf.train.RMSPropOptimizer(0.001)
    ae = CompositeModel(inputs, predict_frames, hidden_num, optimizer=rmsOpti, conditioned=False, targets=targets) 
    print("class %s, hidden_num %d, batch_size %d, epoch %d, optimizer %s, cell %s, learning rate %f, condtioned %s"
            % (type(ae).__name__, hidden_num, batch_size, epoch,
               ae.optimizer, ae.enc_cell, 0.001, ae.conditioned), file=f)
    f.flush()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        train_writer = tf.summary.FileWriter(PATH + 'logdir'+'/train', sess.graph)
        validate_writer = tf.summary.FileWriter(PATH + 'logdir'+'/validate', sess.graph)
        #gradient_writer = tf.summary.FileWriter(PATH + 'logdir'+'/gradient', sess.graph)

        for j in range(epoch):
            for i in range(steps):
                data = data_generator.get_batch().reshape(total_frames, batch_size, -1)
                _, train_sum = sess.run([ae.train, ae.loss_sum],
                                        feed_dict={inputs:data[:10], targets:data[10:]})
            train_writer.add_summary(train_sum, j)
            #gradient_writer.add_summary(gradient_sum, j)

            val_loss_sum = 0
            for p in range(val_steps):
                data = data_generator.get_batch().reshape(total_frames, batch_size, -1)
                val_sum, val_loss = sess.run([ae.loss_sum, ae.loss],
                                             feed_dict={inputs:data[:10], targets:data[10:]})
                val_loss_sum += val_loss
            val_avrg = val_loss_sum/val_steps
            val_summa = tf.Summary(value=[
                tf.Summary.Value(tag="loss", simple_value=val_avrg),])
            validate_writer.add_summary(val_summa, j)
        saver.save(sess, save_path)
        '''
        test_sum = 0
        for k in range(test_steps):
            data = data_generator.get_batch().reshape(total_frames, batch_size, -1)
            test_predict_outputs, test_reconstruct_outputs, test_l = sess.run([ae.predict_outputs, ae.reconstruct_outputs, ae.loss],
                                                                              feed_dict={inputs:data[:10], targets:data[10:]})
            test_sum += test_l
        average_test = test_sum/test_steps
        print("test error %f" % average_test, file=f)

        np.savez_compressed(PATH + "outputs",
                            test_predict_output=test_predict_outputs, test_reconstruct_output= test_reconstruct_outputs,
                            test_in=data)
        '''

    f.close()
    sys.exit(0)
