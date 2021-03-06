import sys
sys.path.insert(0,'..')

import numpy as np
import tensorflow as tf
from onthefly_autoencoder import Autoencoder

DATAPATH = "../../testdata.npz"
MODELPATH = "../../record/model.ckpt"
LOGPATH = "../../record/log.txt"
OUTPUTPATH = "../../record/outputs"

f = open(LOGPATH, "a")
data = np.load(DATAPATH)['test_in']
maxtime = 20 
batch_size = 30 
data = data.reshape(maxtime, batch_size, -1)
desired = data.shape[-1]
hidden_num = 1500

inputs = tf.placeholder(tf.float32,
                        shape = [maxtime, batch_size,
                        desired], name='inputs')
rmsOpti = tf.train.RMSPropOptimizer(0.001)

ae = Autoencoder(inputs, hidden_num, optimizer=rmsOpti, conditioned=False) 

test_steps = 1
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
