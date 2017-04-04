from . import BaseRateBuilder

import numpy as np
from numpy.random.mtrand import _rand as mt
from numba import jit

class LegacyRateBuilder(BaseRateBuilder):
    """
    =====================
      LegacyRateBuilder
    =====================
    
    Introduction
    ============

    This is a rate builder that mimics the behaviour of the rate builder that
    is currently being used in christophs simulation. It simulates a reflecting
    boundary and is thus slow.
    
    Initialization / Parameters
    ===========================

    The following difference equation is simulated
    
    Algorithm Parameters
    --------------------

    *steps_per_ms*
      Integer representing the ms/dt of the current simulation
    
    *channels*
      An iterable representing the channel indices that the builder is
      supposed to build the rate for
    
    *time_length*
      Time length in ms representing the length of the generated rate 
      pattern

    *mean*
      The ou_mean parameter used in the discrete time simulation (in Hz)

    *sigma*
      The ou_sigma parameter used in the discrete time ou simulation (Should
      be in Hz/ms)

    *theta*
      The ou_theta parameter used in the discrete time ou simulation. It
      should be in 1/ms

    *delay*
      The number of TIME STEPS (note. this is NOT INVARIANT to step size) to
      delay before taking the results (allows for burn-in time)

    *max_rate*
      This constitutes a limit on the maximum rate generated and is
      implemented via a reflecting boundary condition. (In Hz)

    Other Parameters
    ----------------

    *rng*
      Specify a random generator for use. If unspecified, defaults to the 
      one used by numpy)

    Algorithm
    =========

    The following difference equation is simulated

        x[n+1] = x[n] + (theta*(mean - x[n]) + sigma*w[n])*(1/steps_per_ms)
        x[n+1] = max_rate if x[n+1] > ln(max_rate)

    The initial conditions are x[0] = mean + N(0,1). The above equation is
    simulated for delay + time_length_in_steps number of steps. Then all
    the values x[delay:] are exponentiated and returned. i.e. exp(x[delay:])
    is returned.

    """

    def __init__(self, config_dict):
        """
        Relevant fields in the config_dict can be seen in the Parameters
        section of the Class documentation
        """
        super().__init__(config_dict)

        self.mean = config_dict['mean']
        self.sigma = config_dict['sigma']
        self.theta = config_dict['theta']
        self.delay = config_dict.get('delay') or 50
        self.max_rate = config_dict.get('max_rate') or 50.0
        self.rng = config_dict.get('rng') or mt

    @staticmethod
    @jit(nopython=True, cache=True)
    def _fast_build(std_normal_array, sigma, mean, theta, steps_per_ms, log_max_rate):
        x = np.zeros_like(std_normal_array, np.float64)
        x[:, 0] = mean + std_normal_array[:,0]
        sim_length = std_normal_array.shape[1]
        for i in range(sim_length-1):
            x[:, i+1] = x[:, i] + (theta*(mean - x[:, i]) + std_normal_array[:,i+1]*sigma)/steps_per_ms
            x[x[:, i+1] > log_max_rate, i+1] = log_max_rate

        return x

    def _build(self):
        nchannels = self._channels.size
        simlength = self._delay + self._steps_length
        sim_array_size = (nchannels, simlength)

        theta = self._theta
        mean = self._mean
        sigma = self._sigma
        log_max_rate = np.log(self._max_rate)
        std_normal_array = self._rng.normal(size=sim_array_size)

        x = LegacyRateBuilder._fast_build(std_normal_array=std_normal_array,
                                          sigma=sigma,
                                          mean=mean,
                                          theta=theta,
                                          steps_per_ms=self._steps_per_ms,
                                          log_max_rate=log_max_rate)
        self._rate_array = np.exp(x[:, self._delay:])


    @property
    def mean(self):
        return self._mean

    @mean.setter
    def mean(self, mean_):
        self._mean = np.float_(mean_)
    

    @property
    def sigma(self):
        return self._sigma

    @sigma.setter
    def sigma(self, sigma_):
        if sigma_ > 0:
            self._sigma = np.float_(sigma_)
        else:
            raise ValueError("'sigma' must be positive")
    

    @property
    def theta(self):
        return self._theta

    @theta.setter
    def theta(self, theta_):
        if self._steps_per_ms > theta_ > 0:
            self._theta = np.float_(theta_)
        else:
            raise ValueError("'theta' must be between 0 and {} for stable filter".format(self._steps_per_ms))
    

    @property
    def delay(self):
        return self._delay

    @delay.setter
    def delay(self, delay_):
        if delay_ > 0:
            self._delay = np.uint32(delay_ + 0.5)
        else:
            raise ValueError("'delay' must be a positive integer")
    

    @property
    def max_rate(self):
        return self._max_rate

    @max_rate.setter
    def max_rate(self, max_rate_):
        if max_rate_ > 0:
            self._max_rate = np.float_(max_rate_)
        else:
            raise ValueError("'max_rate' must be positive")
    

    @property
    def rng(self):
        return self._rng

    @rng.setter
    def rng(self, rng_):
        self._rng = rng_