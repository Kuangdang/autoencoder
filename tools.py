import numpy as np
from new_handler import DataHandler

def normalizedata(data):
    max_e = np.amax(data)
    min_e = np.amin(data)
    norm = 2 * (data - min_e)/(max_e - min_e) - 1
    return norm

def denormdata(data):
    max_e = np.amax(data)
    min_e = np.amin(data)
    norm = 255 * (data - min_e)/(max_e - min_e)
    return norm

def generatedata(dataset, num_frames, save_path, num_digits=2, batch_size=30):
    handler = DataHandler(mnist_dataset=dataset, num_frames=num_frames,
                          num_digits=num_digits, batch_size=batch_size)
    data = handler.get_batch()
    np.savez_compressed(save_path, test_in=data)
