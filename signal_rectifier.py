import matplotlib.pyplot as plt
from scipy.signal import sosfiltfilt, butter
import numpy as np
import math
import argparse

# Helper functions
def round_nearest(num, precision):
    return round(num / precision) * precision

# Signal Rectifier class
class SignalRectifier(object):
    x_data = np.array([])              # original input time data
    y_data = np.array([])              # original input signal data
    clean_y_data = np.array([])        # signal after filtering and smoothening
    jumps = []                         # jump locations on x axis
    gauss_height_lvls = [0]            # gaussian offsets on y axis
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
        #self.print_original_data()

    # print the dataset which was extracted from the input file
    def print_original_data(self):
        for i in range(self.x_data.size):
            print(f"{self.x_data[i]} {self.y_data[i]}")

    # print filtered clean ndataset
    def print_clean_data(self):
        for i in range(self.x_data.size):
            print(f"{self.x_data[i]} {self.clean_y_data[i]}")

    # Plot original dataset
    def plot_original(self):
        plt.plot(self.x_data, self.y_data, label="y_data(x)")
        plt.ylabel('Signal')
        plt.xlabel('Time')
        plt.show()

    # Plot filtered dataset
    def plot_smooth(self):
        try:
            plt.plot(self.x_data, self.clean_y_data, label="clean_y_data(x)")
            plt.ylabel('Signal')
            plt.xlabel('Time')
            plt.show()
        except ValueError:
            print("Error: there is no filtered data")

    # forward-backward digital filter using cascaded second-order sections
    def remove_noise_sosfiltfilt(self, filter_order = 2, critical_freq = 0.05):
        sos = butter(filter_order, critical_freq, output='sos')
        self.clean_y_data = sosfiltfilt(sos, self.y_data)

    # detect x positions of jumps and return an array of those positions
    def detect_jumps(self, sweep = 10, threshold = 0.2):
        '''
        sweep - distance between x samples to compare
        threshold - minimum difference between y samples which defines the beginning of the jump
        '''
        for i in range(self.x_data.size - sweep):
            diff = abs(self.clean_y_data[i + sweep] - self.clean_y_data[i])
            diff = np.round(diff, 2)
            if diff > threshold:
                if round_nearest(self.x_data[i], 0.05) not in self.jumps:
                    # print(f"{self.clean_y_data[i + sweep]}   {self.clean_y_data[i]}")
                    # print(f"diff: {diff}   threshold: {threshold}")
                    self.jumps.append(round_nearest(self.x_data[i + sweep], 0.05))
        # print(self.jumps)
        # identify jump heights
        for jump in self.jumps:
            jump_idx = np.searchsorted(self.x_data, jump, side="left")
            mean_val = np.mean(self.clean_y_data[jump_idx : jump_idx + 200])
            self.gauss_height_lvls.append(mean_val)
        return self.jumps

    # fix the jump to become a smooth transition
    def jump_line_rectify(self, jump_width = 70):
        # get subarray which is a sweep from jumps
        smooth_window = []
        for jump in self.jumps:
            # find the index of closest to jump value in x_data
            jump_idx = np.searchsorted(self.x_data, jump, side="left")
            # collect the window of elements to be smoothened
            jump_window = self.clean_y_data[jump_idx - jump_width : jump_idx + jump_width]
            # draw a straight line between first and last element of jump window
            rect_vals = np.linspace(jump_window[0], jump_window[-1], jump_window.size)
            smooth_window.append(rect_vals)
            self.clean_y_data[jump_idx - jump_width : jump_idx + jump_width] = rect_vals
        return smooth_window

    # fix the jump to become a smooth transition
    def jump_sin_rectify(self, jump_width = 70):
        # get subarray which is a sweep from jumps
        smooth_window = []
        for i, jump in enumerate(self.jumps):
            # find the index of closest to jump value in x_data
            jump_idx = np.searchsorted(self.x_data, jump, side="left")
            # collect the window of elements to be smoothened
            x = np.linspace(0, math.pi, jump_width * 2)
            # prepare the amplitude for the sinusoidal signal
            A = (self.gauss_height_lvls[i + 1] - self.gauss_height_lvls[i]) / 2
            # apply sinusoidal signal to desired area to make the waveform smooth
            rect_vals = A * np.sin(x - math.pi / 2) + self.gauss_height_lvls[i + 1] - A
            smooth_window.append(rect_vals)
            self.clean_y_data[jump_idx - jump_width : jump_idx + jump_width] = rect_vals
        return smooth_window

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Modify a dataset for the test')
    parser.add_argument('--path', dest='filepath', required=True, help='path to the dataset file.')

    # Initialise Signal Rectifier class
    sig_rect = SignalRectifier()
    # remove noise with sosfiltfilt
    sig_rect.remove_noise_sosfiltfilt()
    # find the x locations of signal jumps
    sig_rect.detect_jumps()
    #sig_rect.jump_line_rectify()
    sig_rect.jump_sin_rectify()
    #sig_rect.plot_original()
    sig_rect.plot_smooth()
