#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
from scipy.stats import norm
from math import sqrt
import os, sys, argparse

class GenerateTestData(object):
    data_points = np.array([])
    x = np.array([])
    _nb_points_const = 1000

    def __init__(self, start = 0, end = 3, nb_gaussians = 4):
        self.start = start
        self.end = end
        self.nb_gaussians = nb_gaussians

        total_range = end - start
        self.x = np.linspace(0, total_range, self._nb_points_const * nb_gaussians)

        for i in range(nb_gaussians):
            #create gaussian in proper range

            # python2
            #new_gaussian = mlab.normpdf(self.x[i*self._nb_points_const : (i+1) * self._nb_points_const], i*(float(total_range)/float(nb_gaussians))+0.5, 0.05)
            # python3
            new_gaussian = norm.pdf(self.x[i*self._nb_points_const : (i+1) * self._nb_points_const], i*(float(total_range)/float(nb_gaussians))+0.5, 0.05)

            new_gaussian /= np.random.normal(5.0,0.2,1)[0]
            jump = np.array([i*np.random.normal(1.5,0.2,1)[0]]*new_gaussian.shape[0])
            print("this is jump")
            print(jump)
            self.data_points = np.append(self.data_points, new_gaussian + jump)
        #apply noise
        self.apply_noise()

    def apply_noise(self):
        self.data_points += np.random.normal(0, 0.1,
                                             self.data_points.shape[0])

    def plot(self):
        plt.plot(self.x, self.data_points)
        plt.show()

    def save_to_file(self, path):
        formatted_data = ["x   y\n"]
        for x,y in zip(self.x, self.data_points):
            formatted_data += [str(x) + " " + str(y)+"\n"]
        f = open(path, "w")
        f.writelines(formatted_data)
        f.close()

if __name__ == "__main__":
    generate_test = GenerateTestData()

    parser = argparse.ArgumentParser(description='Generate a dataset for the test')
    parser.add_argument('--path', dest='path', help='path to save the dataset to.')
    parser.add_argument('--plot', dest='plot', help='set this option if you want to plot the data', action="store_true")
    parser.set_defaults(plot=False)
    parser.set_defaults(path=None)
    args = parser.parse_args()

    if args.path != None:
        fn = args.path
        if os.path.exists(os.path.dirname(fn)):
            print("saving to file: ",fn)
            generate_test.save_to_file(fn)
        else:
            print("couldn't find the given path: ",fn)

    if args.plot:
        generate_test.plot()
