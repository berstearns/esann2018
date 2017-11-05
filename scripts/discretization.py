#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Usage:
    >>> from sklearn.datasets import load_iris
    >>> data = load_iris()
    >>> obj = disc.Discretization(data.data, {0:7,1:3, 2:10, 3:4})
    >>> X_t = obj.fit_transform()
    >>> sample = X_t[65,:]
    >>> sample
    #array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0,
    #       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0,
    #       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
    #       1, 0, 0, 0, 0, 0, 0, 0])
    >>> data.data[65,:]
    array([ 6.7,  3.1,  4.4,  1.4])
    >>> obj.inverse_transform(sample)
    array([ 6.66863463,  3.054     ,  4.28622542,  1.38881982])
"""
import numpy as np

class Discretization(object):
    def __init__(self, features, granularity_dict={}):
        """Initializer.
        Arguments:
            features (numpy.ndarray): The set of columns to be discretized.
            granularity_dict (dictionary): A dictionary that specifies the number
                of subdivisions per standard deviations, below and above the average.
                By default, all features will be discretized with a granularity of 5
                per std. The key is the feature index on the features argument,
                and the value is its granularity.
        """
        super(Discretization, self).__init__()
        self.features = features
        self.granularity_dict = granularity_dict
        self.statistics = list()
        self.grans = list()
        self.bins = list()

    def fit(self):
        """Get the features to be discretized and obtain the necessary information
        to the discretization process, as listed below:
        mean,
        standard deviation,
        discretization granularity,
        discretization bins.
        """
        if len(self.features.shape) == 1:
            self.features = self.features.reshape(self.features.shape[0],1)

        for idx in range(self.features.shape[1]):

            if idx in self.granularity_dict.keys():
                gran = self.granularity_dict[idx]
                self.grans.append(gran)
            else:
                gran = 5
                self.grans.append(gran)

            mean = self.features[:,idx].mean()
            std = self.features[:,idx].std()
            self.statistics.append([mean, std])

            less = np.arange(-2*std+mean,
                mean-0.2*(std/gran),
                step=(std/gran))

            greater = np.arange(mean+(std/gran), mean+2*std+(0.5*std/gran), step=(std/gran))

            bins = np.concatenate([
            less,
            np.array([ mean ]),
            greater
            ])
            self.bins.append(bins)

    def transform(self):
        """Do the binary encoding of features.
        Return:
            A numpy.ndarray containing the samples with discretized features.
        """
        discr_final = list()
        for idx in range(self.features.shape[1]):
            mean = self.statistics[idx][0]
            std = self.statistics[idx][1]
            gran = self.grans[idx]
            bins = self.bins[idx]
            middle = int((bins.shape[0]-1)/2)

            discr_samples = list()

            for sample in self.features[:,idx]:
                vec = np.array([False]*bins.shape[0])
                vec[middle] = True

                if sample >= mean:
                    vec[middle+1:] = sample >= bins[middle+1:]
                else:
                    vec[:middle] = sample <= bins[:middle]

                vec = vec.astype(int)
                discr_samples.append(vec.reshape(1, vec.shape[0]))

            feat_discr = np.concatenate(discr_samples,axis=0)
            discr_final.append(feat_discr)

        X_transform = np.concatenate(discr_final, axis=1)

        return X_transform


    def fit_transform(self):
        """Extract the information of features needed to discretization,
        and discretize the features.
        """
        self.fit()
        X = self.transform()
        return X

    def inverse_transform(self, sample):
        """Given a discretization, returns an approximate real value associated with this
        discretization (binary decoding).
        Argument:
            sample (numpy.ndarray): binary discretization.
        Return:
            A numpy.ndarray, which is an approximation to the instance that
                generates the discretization.
        """
        instance = list()
        nbins_slice = [list_of_thresholds.shape[0] for list_of_thresholds in self.bins]
        slices_list = list()
        for nbins in nbins_slice:
            idxs = np.arange(nbins)
            slices_list.append(sample[idxs])
            sample = np.delete(sample, idxs)
        for idx in range(len(slices_list)):
            middle = (nbins_slice[idx]-1)/2
            if slices_list[idx][int(middle+1)] == 1:
                stds = np.where(slices_list[idx] == 1)[0][-1] - middle

            elif slices_list[idx][int(middle-1)] == 1:
                stds = np.where(slices_list[idx] == 1)[0][0] - middle

            else:
                stds = 0

            mean = self.statistics[idx][0]
            std = self.statistics[idx][1]
            gran = self.grans[idx]
            aprox_value = mean + (stds)*float(std/gran)
            instance.append(aprox_value)

        return np.array(instance)

    def get_slices_from_discretized_sample(self, sample):
        """Helper function to get feature slices from a discretized sample.
        Arguments:
            sample (numpy.ndarray): the discretized sample.
        Return:
            A numpy.ndarray with the discretizations of each feature separately.
        """
        slices = [e.shape[0] for e in self.bins]
        feature_slices = list()
        for sli in slices:
            idxs = np.arange(sli)
            feature_slices.append(sample[idxs])
            sample = np.delete(sample, idxs)
        return np.array(feature_slices)
