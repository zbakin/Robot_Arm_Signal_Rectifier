import matplotlib.pyplot as plt
from scipy.signal import sosfiltfilt, butter
import numpy as np
import math, argparse, sys, os
import warnings

# Helper functions
def round_nearest(num, precision):
    return round(num / precision) * precision

# Signal Rectifier class
class SignalRectifier(object):
    # Initialise the signal rectifier object
    def __init__(self, filename):
        self.x_data = np.array([])              # original input time data
        self.y_data = np.array([])              # original input signal data
        self.clean_y_data = np.array([])        # signal after filtering and smoothening
        self.jump_points = []                   # jump locations on x axis
        self.gauss_height_lvls = [0]            # gaussian offsets on y axis
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

    # print the dataset which was extracted from the input file
    def print_original(self):
        print("# Dump of original dataset ###########")
        for i in range(self.x_data.size):
            print(f"{self.x_data[i]} {self.y_data[i]}")

    # print filtered clean ndataset
    def print_clean(self):
        print("# Dump of clean dataset ###########")
        for i in range(self.x_data.size):
            print(f"{self.x_data[i]} {self.clean_y_data[i]}")

    # Plot original dataset
    def plot_original(self):
        plt.plot(self.x_data, self.y_data, label="original signal")
        plt.ylabel('Signal')
        plt.xlabel('Time')
        plt.legend()
        plt.show()

    # Plot filtered dataset
    def plot_smooth(self):
        try:
            plt.plot(self.x_data, self.clean_y_data, label="rectified signal")
            plt.ylabel('Signal')
            plt.xlabel('Time')
            plt.legend()
            plt.show()
        except ValueError:
            print("Error: there is no filtered data generated")

    # Plot original and smooth data on 1 graph
    def plot(self):
        try:
            plt.plot(self.x_data, self.y_data, label="original signal")
            plt.plot(self.x_data, self.clean_y_data, label="rectified signal")
            plt.ylabel('Signal')
            plt.xlabel('Time')
            plt.legend()
            plt.show()
        except ValueError:
            print("Error: there is no filtered data generated")

    # forward-backward digital filter using cascaded second-order sections
    def remove_noise_sosfiltfilt(self, filter_order = 2, critical_freq = 0.05):
        sos = butter(filter_order, critical_freq, output='sos')
        self.clean_y_data = sosfiltfilt(sos, self.y_data)

    # detect x positions of jumps and return an array of those positions
    def detect_jumps(self, sweep = 10, threshold = 0.2):
        '''
        sweep - x axis distance between samples to compare
        threshold - minimum difference between y samples which defines the beginning of the jump
        '''
        for i in range(self.x_data.size - sweep):
            diff = abs(self.clean_y_data[i + sweep] - self.clean_y_data[i]) # difference between 2 samples of "sweep" distance
            diff = np.round(diff, 2)
            if diff > threshold:
                if round_nearest(self.x_data[i], 0.05) not in self.jump_points:
                    # store the jump locations
                    self.jump_points.append(round_nearest(self.x_data[i + sweep], 0.05))
        # Add warning suppress when mean of empty array is calculated
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            # identify jump heights
            flat_region_samples = 400   # increase this to get maximum precision up until around 500(where the trajectory starts to bend)
            for jump in self.jump_points:
                jump_idx = np.searchsorted(self.x_data, jump, side="left")
                # get signal height level of the guassian
                mean_val = np.mean(self.clean_y_data[jump_idx : jump_idx + flat_region_samples])
                self.gauss_height_lvls.append(mean_val)
        return self.jump_points

    # fix the jump to get a smooth transition
    def jump_line_rectify(self, jump_width = 70):
        '''
            Modify the region of the jump to become smooth by connecting first and last points of the transition region
            jump_width - number of samples on each side from the jump point.
                Total number of samples in the area to modify is twice jump_width
        '''
        # Check if there are jump points in the data
        if self.jump_points == []:
            print("INFO: No jump points were identified")
            return
        # array to store modified window datasets
        smooth_windows = []
        for jump in self.jump_points:
            # find the index of closest to jump value in x_data
            jump_idx = np.searchsorted(self.x_data, jump, side="left")
            # collect the window of elements to be smoothened
            jump_window = self.clean_y_data[jump_idx - jump_width : jump_idx + jump_width]
            # draw a straight line between first and last element of jump window
            rect_vals = np.linspace(jump_window[0], jump_window[-1], jump_window.size)
            smooth_windows.append(rect_vals)
            self.clean_y_data[jump_idx - jump_width : jump_idx + jump_width] = rect_vals
        return smooth_windows

    # fix the jump to become a smooth transition
    def jump_sin_rectify(self, jump_width = 70):
        '''
            Modify the region of the jump to become smooth by applying sin wave to the transition region
            jump_width - number of samples on each side from the jump point.
                Total number of samples in the area to modify is twice jump_width
        '''
        # Check if there are jump points in the data
        if self.jump_points == []:
            print("INFO: No jump points were identified")
            return
        # array to store modified window datasets
        smooth_windows = []
        for i, jump in enumerate(self.jump_points):
            # find the index of closest to jump value in x_data
            jump_idx = np.searchsorted(self.x_data, jump, side="left")
            # collect the window of elements to be smoothened
            x = np.linspace(0, math.pi, jump_width * 2)
            # prepare the amplitude for the sinusoidal signal
            A = (self.gauss_height_lvls[i + 1] - self.gauss_height_lvls[i]) / 2
            # apply sinusoidal signal to desired area to make the waveform smooth
            smooth_jump = A * np.sin(x - math.pi / 2) + self.gauss_height_lvls[i + 1] - A
            smooth_windows.append(smooth_jump)
            self.clean_y_data[jump_idx - jump_width : jump_idx + jump_width] = smooth_jump
        return smooth_windows

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Modify a dataset for the test')
    parser.add_argument('--path', dest='filepath', required=True, help='path to the dataset file.')
    parser.add_argument('--plot_original', dest='plot_original', help='set this option if you want to plot original data', action="store_true")
    parser.add_argument('--plot_clean', dest='plot_clean', help='set this option if you want to plot rectified smooth clean data', action="store_true")
    parser.add_argument('--print_original', dest='print_original', help='set this option if you want to print original dataset', action="store_true")
    parser.add_argument('--print_clean', dest='print_clean', help='set this option if you want to print clean dataset', action="store_true")
    parser.add_argument('--simple', dest='simple_rectify', help='set this option if you want a simple rectification. Default is advanced.', action="store_true")

    args = parser.parse_args()
    fn = args.filepath
    # Check if the specified file exists
    if not os.path.exists(fn):
        sys.exit(f"Error: filename {fn} is not available.")

    # Initialise Signal Rectifier class
    SR = SignalRectifier(fn)

    # Print original data
    if args.print_original:
        SR.print_original()

    # remove noise with sosfiltfilt
    SR.remove_noise_sosfiltfilt()

    # find jump points
    SR.detect_jumps()

    if args.simple_rectify:
        # Simplistic rectification
        SR.jump_line_rectify()
    else:
        # Advanced rectification
        SR.jump_sin_rectify()

    # Print clean results
    if args.print_clean:
        SR.print_clean()

    # Plot original dataset
    if args.plot_original:
        SR.plot_original()
    # Plot rectified dataset
    elif args.plot_clean:
        SR.plot_smooth()
    # Plot original and rectified dataset together
    else:
        SR.plot()
