import sys
sys.path.insert(0,'..')

import numpy as np
import tensorflow as tf
from compositemodel import CompositeModel

DATAPATH = "../../testdata.npz"
MODELPATH = "../../record/model.ckpt"
LOGPATH = "../../record/log"
OUTPUTPATH = "../../record/outputs"

f = open(LOGPATH, "w+")
data = np.load(DATAPATH)['test_in']
data = data.reshape(maxtime, batch_size, -1)
input_frames = 10
predict_frames = 10
batch_size = data.shape[1]
desired = data.shape[-1]
hidden_num = 1500

inputs = tf.placeholder(tf.float32,
                        shape = [input_frames, batch_size,
                        desired, name='inputs')

targets = tf.placeholder(tf.float32,
                        shape = [predict_frames, batch_size,
                        desired, name='targets')
rmsOpti = tf.train.RMSPropOptimizer(0.001)

ae = CompositeModel(inputs, predict_frames, hidden_num,
               optimizer=rmsOpti, conditioned=False,
               targets=targets) 

test_steps = 50
saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, MODELPATH)
    test_sum = 0
    for _ in range(test_steps):
        test_outputs, test_l = sess.run([ae.outputs, ae.loss], 
            feed_dict={inputs:data[:10], targets=data[10:]})
        test_sum += test_l 
    average_test = test_sum/test_steps
    f.write("test error %f" % average_test)
    np.savez_compressed(OUTPUTPATH,
                        test_out=test_outputs)
f.close()
