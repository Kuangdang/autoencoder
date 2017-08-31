from __future__ import division

import h5py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from matplotlib import animation
from IPython.core.debugger import Tracer
debug_here = Tracer()



class DataHandler(object):
    """ The data handler that creates Bouncing MNIST dataset on the fly. """

    def __init__(self, mnist_dataset='../data_sets/mnist/mnist.h5.1', mode='standard',
                 background='zeros', num_frames=40, batch_size=2, image_size=64,
                 num_digits=2, step_length=0.1):
        """
        Create a new data Handler. Using the mnist data set from
        http://www.cs.toronto.edu/~emansim/datasets/mnist.h5 is recommendet.
        """
        self.mode_ = mode
        self.background_ = background
        self.seq_length_ = num_frames
        self.batch_size_ = batch_size
        self.image_size_ = image_size
        self.num_digits_ = num_digits
        self.step_length_ = step_length
        self.digit_size_ = 28
        self.frame_size_ = self.image_size_ ** 2
        self.num_channels_ = 1
        f = h5py.File(mnist_dataset)

        self.data_ = f['train'].value.reshape(-1, 28, 28)
        f.close()
        # if self.binarize_:
        #     self.data_ = np.round(self.data_)
        self.indices_ = np.arange(self.data_.shape[0])
        self.row_ = 0
        np.random.shuffle(self.indices_)

    def get_batch_size(self):
        return self.batch_size_

    def get_dims(self):
        return self.frame_size_

    def get_seq_length(self):
        return self.seq_length_

    def get_batch_shape(self):
        return [self.batch_size_, self.seq_length_,
                self.image_size_, self.image_size_, 1]

    def reset(self):
        pass

    def get_random_trajectory(self, batch_size):
        length = self.seq_length_
        canvas_size = self.image_size_ - self.digit_size_

        # Initial position uniform random inside the box.
        y = np.random.rand(batch_size)
        x = np.random.rand(batch_size)

        # Choose a random velocity.
        theta = np.random.rand(batch_size) * 2 * np.pi
        v_y = np.sin(theta)
        v_x = np.cos(theta)

        start_y = np.zeros((length, batch_size))
        start_x = np.zeros((length, batch_size))
        for i in range(length):
            # Take a step along velocity.
            y += v_y * self.step_length_
            x += v_x * self.step_length_

            # Bounce off edges.
            for j in range(batch_size):
                if x[j] <= 0:
                    x[j] = 0
                    v_x[j] = -v_x[j]
                if x[j] >= 1.0:
                    x[j] = 1.0
                    v_x[j] = -v_x[j]
                if y[j] <= 0:
                    y[j] = 0
                    v_y[j] = -v_y[j]
                if y[j] >= 1.0:
                    y[j] = 1.0
                    v_y[j] = -v_y[j]
            start_y[i, :] = y
            start_x[i, :] = x

        # Scale to the size of the canvas.
        start_y = (canvas_size * start_y).astype(np.int32)
        start_x = (canvas_size * start_x).astype(np.int32)
        return start_y, start_x

    def overlap(self, a, b):
        """ Put b on top of a."""
        return np.maximum(a, b)

    def get_batch(self, verbose=False):
        start_y, start_x = self.get_random_trajectory(self.batch_size_ * self.num_digits_)
        data = np.zeros((self.seq_length_, self.batch_size_, self.image_size_,
                         self.image_size_), dtype=np.float32)
        for j in range(self.batch_size_):
            for n in range(self.num_digits_):
                ind = self.indices_[self.row_]
                self.row_ += 1
                if self.row_ == self.data_.shape[0]:
                    self.row_ = 0
                    np.random.shuffle(self.indices_)
                digit_image = self.data_[ind, :, :]
                for i in range(self.seq_length_):
                    top = start_y[i, j * self.num_digits_ + n]
                    left = start_x[i, j * self.num_digits_ + n]
                    bottom = top + self.digit_size_
                    right = left + self.digit_size_
                    data[i, j, top:bottom, left:right] =  \
                        self.overlap(data[i, j, top:bottom, left:right], digit_image)

        # return batch_major data
        #data = np.transpose(data, [1, 0, 2, 3])
        #data = np.expand_dims(data, -1)
        return data


def play_single_video(np_video, save=False, path=None):
    """Play a single numpy array of shape [time, hight, width, 3]
       as RGB video.
    """
    # np_video = load_video(path)
    frame_no = np_video.shape[0]

    fig = plt.figure()  # make figure
    im = plt.imshow(np_video[0, :, :, 0])

    # function to update figure
    def updatefig(j):
        # set the data in the axesimage object
        im.set_array(np_video[j, :, :, 0])
        # return the artists set
        return im,

    # kick off the animation
    anim = animation.FuncAnimation(fig, updatefig, frames=range(frame_no),
                                   interval=50, blit=True)
    if save is True:
        if path is None:
            print("missing path. Saving to result.mp4 and result.pkl")
            path = 'result'
        anim.save(path + '.mp4', fps=10, writer="avconv", codec="libx264")
        import pickle
        pickle.dump(np_video, open(path + '.pkl', 'wb'))

    plt.show()
