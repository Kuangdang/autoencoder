import sys
sys.path.insert(0,'..')

import numpy as np
import tensorflow as tf
from convautoencoder import ConvAutoencoder
from custom_cell import ConvLSTMCell

DATAPATH = "../../testdata.npz"
MODELPATH = "../../record/model.ckpt"
LOGPATH = "../../record/log"
OUTPUTPATH = "../../record/outputs"
f = open(LOGPATH, "w+")
data = np.load(DATAPATH)['test_in']
maxtime = data.shape[0]
batch_size = data.shape[1]
in_h = data.shape[2]
in_w = data.shape[3]
enc_cell = ConvLSTMCell(30, (in_h, in_w), [8,8], 1)
dec_cell = ConvLSTMCell(30, (in_h, in_w), [8,8], 1)
inputs = tf.placeholder(tf.float32,
                        shape = [maxtime, batch_size,
                                 in_h, in_w, 1], name='inputs')
rmsOpti = tf.train.RMSPropOptimizer(0.001)
ae = ConvAutoencoder(inputs, enc_cell=enc_cell,
                     dec_cell=dec_cell, optimizer=rmsOpti,
                     conditioned=False) 

data = data.reshape(maxtime, batch_size, in_h, in_w, 1)
test_steps = 50
saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, MODELPATH) 
    test_sum = 0
    for _ in range(test_steps):
        test_outputs, test_l = sess.run([ae.outputs, ae.loss], 
                                        feed_dict={inputs:data})
        test_sum += test_l 
    average_test = test_sum/test_steps
    f.write("test error %f" % average_test)
    np.savez_compressed(OUTPUTPATH,
                        test_out=test_outputs)
f.close()
