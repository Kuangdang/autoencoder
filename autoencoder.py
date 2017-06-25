import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
import tensorflow as tf
import sys


f = open("/home/stud/wangc/lab/record/log", "w+")

data = np.load('/home/stud/wangc/lab/mnist_test_seq.npy')
print(data.shape, file=f)
print(data[0,0].shape, file=f)
#plt.imshow(data[0,0,:,:])
data = np.around(data/255, decimals=5)
data = data.reshape(data.shape[0],data.shape[1],-1)
print(data.shape, file=f)

hidden_num = 2000
maxtime = 20
batch_size = 50
frame_size = 64
desired = frame_size*frame_size

learning_rate_start = 0.001
graph = tf.Graph()
with graph.as_default():
    enc_inputs = tf.placeholder(tf.float32, shape = [maxtime, batch_size, desired], name='enc_inputs')

    with tf.variable_scope('encode', reuse=None):
        enc_cell = tf.contrib.rnn.LSTMCell(hidden_num, use_peepholes=True)
        _, enc_state = tf.nn.dynamic_rnn(enc_cell, enc_inputs, dtype=tf.float32, time_major=True)
        print('enc_state %r' % (enc_state,), file=f)


    saved_output = tf.Variable(tf.zeros([batch_size, hidden_num]), trainable=False)
    saved_state = tf.Variable(tf.zeros([batch_size, hidden_num]), trainable=False)

    # projection weights and biases.
    w = tf.Variable(tf.truncated_normal([hidden_num, desired], -0.1, 0.1, dtype=tf.float32), name='w')
    b = tf.Variable(tf.zeros([desired]),name='b', dtype=tf.float32)

    outputs = list()
    output = saved_output
    state = saved_state
    dec_inputs = tf.zeros([batch_size, desired])
    dec_state = enc_state

    with tf.variable_scope('dec_scope',reuse=None):
        dec_cell = tf.contrib.rnn.LSTMCell(hidden_num, use_peepholes=True)
        v = None
        for _ in range(maxtime):
            with tf.variable_scope('unrolling', reuse=v):
                output, dec_state = dec_cell(dec_inputs, dec_state)
                if v == None:
                    print('output shape %r' % (output.shape,), file=f)
                v = True
                outputs.append(output)   # State saving across unrollings.
    with tf.control_dependencies([saved_output.assign(output),
                                    saved_state.assign(state)]):
        print('before stack shape %r' % (outputs[0].shape), file=f)       
        outputs = [tf.matmul(i, w) + b for i in outputs]
        outputs = tf.stack(outputs)
        print('projected shape %r' % (outputs.shape,), file=f)
        reversed_outputs = outputs[::-1]
        print('reversed shape %r' % (reversed_outputs.shape,), file=f)
        loss = tf.reduce_mean(tf.squared_difference(reversed_outputs,enc_inputs))
    # Optimizer.
    '''
    global_step = tf.Variable(0, trainable = False)
    learning_rate = tf.train.exponential_decay(
            learning_rate_start, global_step, 3000, 0.9, staircase=True)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    gradients, v = zip(*optimizer.compute_gradients(loss))
    gradients, _ = tf.clip_by_global_norm(gradients, 1.25)
    optimizer = optimizer.apply_gradients(
            zip(gradients, v), global_step=global_step)
    '''
    # RMSProp
    '''
    global_step = tf.Variable(0, trainable = False)
    optimizer = tf.train.RMSPropOptimizer(learning_rate_start)
    gradients, v = zip(*optimizer.compute_gradients(loss))
    gradients, _ = tf.clip_by_global_norm(gradients, 10)
    optimizer = optimizer.apply_gradients(
            zip(gradients, v), global_step=global_step)
    '''
    #Adam
    global_step = tf.Variable(0, trainable = False)
    optimizer = tf.train.AdamOptimizer(learning_rate_start)
    gradients, v = zip(*optimizer.compute_gradients(loss))
    gradients, _ = tf.clip_by_global_norm(gradients, 5)
    optimizer = optimizer.apply_gradients(
            zip(gradients, v), global_step=global_step)


    #saver = tf.train.Saver()
    #create summary for loss
    loss_sum = tf.summary.scalar('loss', loss)
    #train_loss = tf.summary.scalar('train_loss', loss)
    #val_loss = tf.summary.scalar("val_loss", loss)
    #test_loss = tf.summary.scalar("test_loss", loss)
    #tf.summary.tensor_summary('gradients', gradients)
    #tf.summary.tensor_summary('projection_weights', w)
train_size = 9000
epoch = 200
steps = int(train_size/batch_size)

#restore = False
with tf.Session(graph=graph) as session:
    #if restore==False:
        tf.global_variables_initializer().run()
        train_writer = tf.summary.FileWriter('/home/stud/wangc/lab/record/logdir'+'/train', graph)
        validate_writer = tf.summary.FileWriter('/home/stud/wangc/lab/record/logdir'+'/validate', graph)
        print('Initialized', file=f)
        print("hidden_num %d, batch_size %d, learning_rate_start %f, epoch %d" % (hidden_num, batch_size, learning_rate_start, epoch), file=f)
        f.flush()
        val_outputs = list()
        for j in range(epoch):
            for i in range(steps):
                _, train_sum = session.run([optimizer, loss_sum], feed_dict={enc_inputs:(data[:,i*batch_size:(i+1)*batch_size])})
            train_writer.add_summary(train_sum, j)
            val_output, val_sum = session.run([outputs, loss_sum], feed_dict={enc_inputs:(data[:,9000:9000+batch_size])})
            validate_writer.add_summary(val_sum, j)
            if j%20 == 0:
                val_outputs.append(val_output)  
        val_outputs = np.array(val_outputs)
        #saver.save(session,"/home/stud/wangc/lab/record/model.ckpt")
        train_writer.close()
        validate_writer.close()
    #else:
        #saver.restore(session, "./model.ckpt")
        #print("model restored")
        test_outputs, test_l = session.run([outputs, loss], feed_dict={enc_inputs:data[:,9500:9500+batch_size]})
        print("test error %f" % test_l, file=f)
        np.savez_compressed("/home/stud/wangc/lab/record/outputs", val_out=val_outputs, val_in=data[:,9000:9000+batch_size], test_out=test_outputs, test_in=data[:,9500:9500+batch_size])
        print('finished', file=f)
f.close()
sys.exit(0)
