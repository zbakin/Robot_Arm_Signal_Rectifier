import matplotlib.pyplot as plt
from scipy.signal import lfilter, sosfiltfilt, butter
import numpy as np
from scipy.stats import norm
import argparse

# Helper functions
def round_nearest(num, precision):
    return round(num / precision) * precision

# Signal Rectifier class
class SignalRectifier(object):
    x_data = np.array([])
    y_data = np.array([])
    rect_y_data = np.array([])
    # Initialise the signal rectifier object
    def __init__(self):
        args = parser.parse_args()
        filename = args.filepath
        self.read_file(filename)

    # read file with input data from the motion capture system
    def read_file(self, filename):
        with open(filename) as file:
            # ignore the first header line
            next(file)
            # iterate through every line and append the values
            for line in file:
                x_and_y = line.split()
                self.x_data = np.append(self.x_data, float(x_and_y[0]))
                self.y_data = np.append(self.y_data, float(x_and_y[1]))
        #self.print_data()

    # print the dataset which was extracted from the input file
    def print_data(self):
        for i in range(self.x_data.size):
            #print(f"{self.x_data[i]} {self.y_data[i]}")
            print(f"{self.x_data[i]} {self.rect_y_data[i]}")

    # Plot the dataset
    def plot(self):
        #plt.plot(self.x_data, self.y_data, label="y_data(x)")
        plt.plot(self.x_data, self.rect_y_data, label="rect_y_data(x)")
        plt.ylabel('Signal')
        plt.xlabel('Time')
        plt.show()

    # removes noise from input signal
    def remove_noise_lfilter(self, smoothness = 15):
        n = smoothness  # the larger n is, the smoother curve will be
        b = [1.0 / n] * n
        a = 1
        self.rect_y_data = lfilter(b, a, self.y_data)
    
    # more advanced filtering - forward-backward digital filter using cascaded second-order sections
    def remove_noise_sosfiltfilt(self, filter_order = 4, critical_freq = 0.08):
        # by default - lowpass Butterworth filter
        sos = butter(filter_order, critical_freq, output='sos')
        self.rect_y_data = sosfiltfilt(sos, self.y_data)

    # detect x positions of jumps and return an array of those positions
    '''
        sweep - distance between x samples to compare
        threshold - minimum difference between y samples which defines the beginning of the jump
    '''
    def detect_jumps(self, sweep = 10, threshold = 0.3):
        jumps = []
        for i in range(self.x_data.size - sweep):
            diff = abs(self.rect_y_data[i + sweep] - self.rect_y_data[i])
            diff = np.round(diff, 2)
            if diff > threshold:
                if round_nearest(self.x_data[i], 0.05) not in jumps:
                    print(f"{self.rect_y_data[i + sweep]}   {self.rect_y_data[i]}")
                    print(f"diff: {diff}   threshold: {threshold}")
                    jumps.append(round_nearest(self.x_data[i + sweep], 0.05))
        return jumps

    # fix the jump to become a smooth transition
    def jump_line_rectify(self, jumps, jump_width = 50):
        # get subarray which is a sweep from jumps
        smooth_window = []
        for jump in jumps:
            # find the index of closest to jump value in x_data
            jump_idx = np.searchsorted(self.x_data, jump, side="left")
            # collect the window of elements to be smoothened
            jump_window = self.rect_y_data[jump_idx - jump_width : jump_idx + jump_width]
            # draw a straight line between first and last element of jump window
            rect_vals = np.linspace(jump_window[0], jump_window[-1], jump_window.size)
            smooth_window.append(rect_vals)
            self.rect_y_data[jump_idx - jump_width : jump_idx + jump_width] = rect_vals
        return smooth_window

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Modify a dataset for the test')
    parser.add_argument('--path', dest='filepath', required=True, help='path to the dataset file.')
    # this variable is to store the jump x locations
    jumps = []
    # Initialise Signal Rectifier class
    sig_rect = SignalRectifier()
    # remove noise with sosfiltfilt
    sig_rect.remove_noise_sosfiltfilt()
    # find the x locations of signal jumps
    jumps = sig_rect.detect_jumps()
    print(jumps)
    rect_areas = sig_rect.jump_line_rectify(jumps)
    sig_rect.plot()
