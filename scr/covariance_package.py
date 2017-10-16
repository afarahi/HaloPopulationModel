#!/usr/bin/env python2.7

import numpy as np

from scipy import stats
from sklearn import linear_model


class covariance_package():

    def __init__(self):

        pass


    def scatter_cal(self, x, y, slope, intercept, dof=None, weight=None):

        if dof == None:
            dof = len(x)

        if weight == None:
            sig2 = sum((np.array(y) - (slope*np.array(x)+intercept))**2) / dof
        else:
            sig2 = np.average((np.array(y)-(slope*np.array(x)+intercept))**2, weights=weight)

        return np.sqrt(sig2)


    def _gaussian_filter(self, x, mu=0.0, sig=1.0):
        return np.exp(-(x-mu)**2/2./sig**2)


    def _uniform_filter(self, x, xmin=-1.0, xmax=1.0):
        w = np.zeros(len(x))
        w[(x<xmax)*(x>xmin)] = 1.0
        return w


    def calculate_weigth(self, x, weigth_type='gussian', *args, **kwargs):
        if weigth_type == 'gussian':
             w = self._gaussian_filter(x, **kwargs)
        elif weigth_type == 'uniform':
             w = self._uniform_filter(x, **kwargs)
        else:
             print "Warning : ", weigth_type, "is not a defined filter."
             print "It assumes w = 1 for every point."
             w = np.ones(len(x))
        return w
             

    def linear_regression(self, x, y, weight=None):

        if weight == None:
            slope, intercept, r_value, p_value, _ = stats.linregress(x, y)
        else:
            regr = linear_model.LinearRegression()
            # Train the model using the training sets
            regr.fit(x[:, np.newaxis], y, sample_weight=weight)
            slope = regr.coef_[0]
            intercept = regr.intercept_
        #sig = self.scatter_cal(x, y, slope, intercept, len(x)-2)
        sig = self.scatter_cal(x, y, slope, intercept, weight=weight)
        return intercept, slope, sig


    def calc_correlation_fixed_x(self, x, y, z, weight=None):

        intercept, slope, sig = self.linear_regression(x, y, weight=weight)
        dy = y - slope * x - intercept
        intercept, slope, sig = self.linear_regression(x, z, weight=weight)
        dz = z - slope * x - intercept
        if weight == None:
            sig = np.cov(dy, dz)
        else:
            sig = np.cov(dy, dz, aweights=weight)
        return sig[1,0] / np.sqrt(sig[0,0]*sig[1,1])

        
        
    #def calculate_universal_correlation(self, xLine, x, y, z,)


""" TEST
import numpy.random as npr
#npr.seed(800)

ndata = 800
x = np.linspace(-5.,5.,ndata)
ran1 = npr.normal(0.0,0.5,ndata)
ran2 = npr.normal(0.0,0.1,ndata)
y = x*1.2 - 0.4 + ran1 + ran2
z = x*0.8 + 0.2 + ran2

cov = covariance_package()
w = cov.calculate_weigth(weigth_type='gussian', mu=3.0, sig = 0.5)


#print cov.linear_regression(x, y, weight=None)
#print cov.linear_regression(x, z, weight=None)
cov.calc_correlation_fixed_x(x, y, z, weight=None)


#print cov.linear_regression(x, y, weight=w)
#print cov.linear_regression(x, z, weight=w)
cov.calc_correlation_fixed_x(x, y, z, weight=w)
"""














