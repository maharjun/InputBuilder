__author__ = 'Arjun'

from . import BaseRateBuilder
from genericbuilder.propdecorators import *

import numpy as np
import numpy.random as rd
import scipy.signal as sg
from collections import namedtuple

class OURateBuilder(BaseRateBuilder):
    """Rate Generator via Homogenous OU Process
    
    This builder generates the rate array based on a discrete approximation of a continuous 
    time OU Process. The approximation matches the rd[0] and rd[1] of the discrete AR1
    process representing the OU process, with the rc(0) and rc(time_step) of the continuous
    time process. The parameters specified are the parameters for the continuous time process.

    Parameters for the OU Process (Note that all these parameters MUST be specified)

    1.  mean  - [Hz]   The mean rate generated by the CT-OU Process
    2.  sigma - [Hz]   The scaling of the Wiener Process
    3.  theta - [1/ms] The rate of decay of OU Process towards mean

    The differential equation is

        dx = theta(mu - x)dt + sigma*dW

    where W is a wiener process with variance given as
        
        var(W(t+h) - W(t)) = alpha h
        with alpha = 1/ms

    The following variables are the discrete time (DT) equivs (with h [ms] as the
    generation time step)
    
    1.  mean_DT  - [Hz]
    2.  sigma_DT - [Hz/ms]
    3.  theta_DT - [1/ms]

    In the fllowing difference equation.

        x[n] - x[n-1] = (theta_DT(mean_DT - x[n-1]) + sigma_DT*w[n])*h
        Where w[n] is AWGN with std-dev = 1

    The following equations relating the DT and CT correltions are used to perform
    the conversion. Note that the mean remains unchanged.

        rc(0) = sigma^2*alpha/(2*theta) = sigma_DT^2 h/(theta_DT(2 - theta_DT*h)) = rd[0]

        rc(h) = rc(0)e^(-theta*h) = rd[0]*(1-theta_DT*h) = rd[1]
    """

    def __init__(self, conf_dict):

        super(OURateBuilder, self).__init__(conf_dict)

        # Setting OU Parameters
        self.mean     = conf_dict['mean']
        self.sigma    = conf_dict['sigma']
        self.theta    = conf_dict['theta']


    def _preprocess(self):
        super()._preprocess()
        self._DT_params = self.convert_params_CT_to_DT()

    def convert_params_CT_to_DT(self):
        """Performs conversion between CT and DT Params"""
        mean = self._mean
        sigma = self._sigma
        theta = self._theta
        h = 1/self._steps_per_ms

        mean_DT  = mean
        theta_DT = (1 - np.exp(-theta*h))/h
        sigma_DT = sigma*np.sqrt(theta_DT * (2 - theta_DT*h)/(2*h*theta))  # ignoring alpha = 1
        DT_params = namedtuple('DT_param_struct', ['mean', 'theta', 'sigma'])(
            mean=mean_DT,
            theta=theta_DT,
            sigma=sigma_DT)
        return DT_params


    @property
    def mean(self):
        return self._mean

    @mean.setter
    @requires_rebuild
    @requires_preprocessing
    def mean(self, mean_):
        self._mean = np.float_(mean_)


    @property
    def sigma(self):
        return self._sigma

    @sigma.setter
    @requires_preprocessing
    @requires_rebuild
    def sigma(self, sigma_):
        if sigma_ > 0:
            self._sigma = np.float_(sigma_)
        else:
            raise ValueError("'sigma' must have a positive non-zero value")


    @property
    def theta(self):
        return self._theta

    @theta.setter
    @requires_rebuild
    @requires_preprocessing
    def theta(self, theta_):
        if theta_ > 0:
            self._theta = np.float_(theta_)
        else:
            raise ValueError("'theta' must have a positive non-zero value")

    @property
    @requires_preprocessed
    def DT_params(self):
        """
        Returns a named tuple with the following fields:

            mean
            sigma
            theta

        representing the corresponding DT parameters
        """
        return self._DT_params


    def _build(self):
        """
        Does the actualwork of generating the OU Process from converted DT OU
        Parameters. This uses the scipy filter to create the process using the
        filter representing the difference equation below

        y[n] - y[n-1] = -(theta_DT*y[n] + sigma_DT*w[n])*h
        [where y[n] = x[n] - mean]

        => Y(z) = H(z)W(z)

        with H(z) = sigma_DT*h/(1 - z^{-1}(1-theta_DT*h))

        In order to remove initial transients we only consider the samples
        n > d such that rd[d] < thresh*rd[0] i.e. only d :: (1-theta_DT*h)^d < thresh
        """        
        super()._build()

        theta_DT = self._DT_params.theta
        sigma_DT = self._DT_params.sigma
        mean     = self._DT_params.mean
        h        = 1/self._steps_per_ms
        
        # # Debug prints
        # print("CT variance: {:<10.5f}".format(self._sigma**2/(2*self._theta)))
        # print("DT variance: {:<10.5f}".format(sigma_DT**2*h/(theta_DT*(2 - theta_DT*h))))
        # print()
        # print("CT Mean: {:<10.5f}".format(self._mean))
        # print("DT Mean: {:<10.5f}".format(mean))
        # print()
        # print("CT Corr: {:<10.5f}".format(np.exp(-self._theta*h)))
        # print("DT Corr: {:<10.5f}".format(1 - theta_DT*h))
        # print()

        # Calculate Shape Vectors
        curr_rate_array_shape = (self._channels.size, self._steps_length)

        # Calculate Initial Condition from Steady state distribution of
        # OU Process. This way we wont have to wait for the process to burn in 
        steady_state_SD = self._sigma/np.sqrt(2*self._theta)
        rate_array_init = np.random.normal(loc=0, scale=steady_state_SD, size=(self._channels.size, 1))
        filter_init_cond = (1- theta_DT*h)*rate_array_init

        awgn_array = rd.normal(size=curr_rate_array_shape)
        filtered_awgn_array, __ = sg.lfilter([sigma_DT*h], [1, -(1 - theta_DT*h)], awgn_array, zi=filter_init_cond)

        self._rate_array = filtered_awgn_array + mean
