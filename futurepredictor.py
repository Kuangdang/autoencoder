import sys
import numpy as np
import tensorflow as tf
from new_handler import DataHandler
#autoencoder class
class Predictor:
    def __init__(self, inputs, predict_frames_num, hidden_num,
                 targets=None, enc_cell=None, dec_cell=None, optimizer=None, conditioned=None):
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
            self.dec_cell = tf.contrib.rnn.LSTMCell(self.hidden_num,
                                                    use_peepholes=True, num_proj=desired)
        else:
            self.enc_cell = enc_cell
            self.dec_cell = dec_cell

        with tf.variable_scope('encode', reuse=None):
            _, enc_state = tf.nn.dynamic_rnn(self.enc_cell,
                                             inputs, dtype=tf.float32, time_major=True)

        #assum projection is performed in cell
        saved_output = tf.Variable(tf.zeros([batch_size, desired]), trainable=False)
        saved_state = tf.Variable(tf.zeros([batch_size, desired]), trainable=False)

        outputs = list()
        output = saved_output
        state = saved_state

        with tf.variable_scope('dec_scope', reuse=None):
            dec_inputs = tf.zeros([batch_size, desired])
            dec_state = enc_state
            v = None
            for _ in range(predict_frames):
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

        if targets is not None:
            self.loss = tf.reduce_mean(tf.squared_difference(self.outputs, targets))
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
    ae = Predictor(inputs, predict_frames, hidden_num, optimizer=rmsOpti, conditioned=False, targets=targets) 
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
            test_outputs, test_l = sess.run([ae.outputs, ae.loss],
                                                feed_dict={inputs:data[:10], targets:data[10:]})
            test_sum += test_l
        average_test = test_sum/test_steps
        print("test error %f" % average_test, file=f)

        np.savez_compressed(PATH + "outputs",
                test_out=test_outputs, test_in=data[10:])
        '''

    f.close()
    sys.exit(0)
