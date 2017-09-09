import numpy as np
from tools import generatedata

dataset = '../mnist.h5'
generatedata(dataset, num_frames=20,
             save_path='../testdata.npz', num_digits=2)
