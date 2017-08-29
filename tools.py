import numpy as np

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
