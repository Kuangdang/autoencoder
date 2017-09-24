import sys
sys.path.insert(0,'..')

import numpy as np
import tensorflow as tf
from convpredictor import ConvPredictor
from custom_cell import ConvLSTMCell

DATAPATH = "../../testdata.npz"
MODELPATH = "../../record/model.ckpt"
LOGPATH = "../../record/log"
OUTPUTPATH = "../../record/outputs"
f = open(LOGPATH, "a")
data = np.load(DATAPATH)['test_in']
input_frames = 10
predict_frames = 10
total_frames = input_frames + predict_frames
batch_size = data.shape[1]
in_h = data.shape[2]
in_w = data.shape[3]
enc_cell = ConvLSTMCell(100, (in_h, in_w), [8,8], 1)
dec_cell = ConvLSTMCell(100, (in_h, in_w), [8,8], 1)
inputs = tf.placeholder(tf.float32, shape = [input_frames, batch_size, in_h, in_w, 1], name='inputs')               
targets = tf.placeholder(tf.float32, shape = [predict_frames, batch_size, in_h, in_w, 1], name='targets')
rmsOpti = tf.train.RMSPropOptimizer(0.0001)
ae = ConvPredictor(inputs, predict_frames=10, enc_cell=enc_cell, dec_cell=dec_cell, targets=targets,
                       optimizer=rmsOpti, conditioned=False) 

data = data.reshape(total_frames, batch_size, in_h, in_w, 1)
#print(data[0,0])
test_steps = 1
saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, MODELPATH) 
    test_sum = 0
    for _ in range(test_steps):
        test_outputs, test_l = sess.run([ae.outputs, ae.loss], 
                                        feed_dict={inputs:data[:10], targets:data[10:]})
        print(test_l)
        test_sum += test_l 
    average_test = test_sum/test_steps
    f.write("test error %f" % average_test)
    np.savez_compressed(OUTPUTPATH,
                        test_out=test_outputs)
f.close()
