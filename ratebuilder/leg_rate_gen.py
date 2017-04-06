from . import BaseRateBuilder

import numpy as np
from numpy.random import mtrand
from numba import jit
mtgen = mtrand.binomial.__self__

from genericbuilder.propdecorators import requires_built

class LegacyRateBuilder(BaseRateBuilder):
    """
    =====================
      LegacyRateBuilder
    =====================
    
    Introduction
    ============

    This is a rate builder that mimics the behaviour of the rate builder that
    is currently being used in christophs simulation. It simulates a reflecting
    boundary and is thus slow without numba.
    
    Initialization / Parameters
    ===========================

    The following difference equation is simulated
    
    Algorithm Parameters
    --------------------

    *steps_per_ms*    
      Integer [1/ms] representing the number of simulation steps ms of the current simulation
    
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

    built_properties = 'rate_array'

    def __init__(self, mean, sigma, theta, delay, max_rate,
                       channels=[], steps_per_ms=1, time_length=0,
                       rng=mtgen):
        """
        Relevant fields in the config_dict can be seen in the Parameters
        section of the Class documentation
        """
        super().__init__()  # only purpose is to run BaseGenericBuilder init

        # initializing Data members with default values
        self._channels = np.zeros(0, dtype=np.uint32)
        self._steps_per_ms = np.uint32(1)
        self._time_length = np.float64(0)
        self._rng = mtgen

        # Assigning core properties
        self.channels = channels
        self.steps_per_ms = steps_per_ms
        self.time_length = time_length

        # Assigning the random generator
        self.rng = rng

        # Assigning the compulsory / positional arguments
        self.mean = mean
        self.sigma = sigma
        self.theta = theta
        self.delay = delay
        self.max_rate = max_rate

    def _preprocess(self):
        # Calculating all the dependent variables
        self._steps_length = int(self._time_length*self._steps_per_ms + 0.5)

    def _validate(self):
        pass
    
    @staticmethod
    @jit(nopython=True, cache=True)
    def _fast_build(std_normal_array, sigma, mean, theta, steps_per_ms, log_max_rate):
        x = np.zeros_like(std_normal_array, np.float64)
        x[:, 0] = mean + std_normal_array[:,0]
        sim_length = std_normal_array.shape[1]
        for i in range(sim_length-1):
            x[:, i+1] = x[:, i] + (theta*(mean - x[:, i]) + std_normal_array[:, i+1]*sigma)/steps_per_ms
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
        self._rate_array.setflags(write=False)


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

    @property
    def steps_per_ms(self):
        """
        Get or Set the time resolution of the rate pattern by specifying an integer representing
        the number of time steps per ms

        :return:
        """
        return self._steps_per_ms

    @steps_per_ms.setter
    def steps_per_ms(self, steps_per_ms_):
        if steps_per_ms_ is None:
            self._init_attr('_steps_per_ms', np.uint32(1))
        else:
            if steps_per_ms_ >= 1:
                self._steps_per_ms = np.uint32(steps_per_ms_)
            else:
                raise ValueError("'steps_per_ms' must be non-zero positive integer")

    @property
    def time_length(self):
        """
        Get or set the time length of the rate pattern in ms.

        :GET: This function will return the time length of the built rate pattern i.e. the
            rounded time length. i.e. `self._steps_length/self.steps_per_ms`

        :SET: This function will round the time to the nearest time step and use that as the actual
            time length.

        :return: An np.float64 scalar
        """
        return self._steps_length / self._steps_per_ms

    @time_length.setter
    def time_length(self, time_length_):
        if time_length_ >= 0:
            self._time_length = np.float64(time_length_)
        else:
            raise ValueError("property 'time_length' must be a non-negative numeric value")

    @property
    def channels(self):
        """
        Returns channels property. In order to make it writable, copy it via X.channels.copy() or
        np.array(X.channels)
        """
        return self._channels

    @channels.setter
    def channels(self, channels_):
        # Assuming 1D iterable
        channel_unique_array = np.array(list(set(channels_)), dtype=np.int32)
        if np.all(channel_unique_array >= 0):
            self._channels = np.array(channel_unique_array, dtype=np.uint32)
            self._channels.setflags(write=False)
        else:
            raise ValueError("'channels' must be a vector of non-negative integers")

    @property
    @requires_built
    def rate_array(self):
        """
        Returns read-only view of channels property. In order to make it writable, copy it
        via X.rate_array.copy() or np.array(X.rate_array)
        """
        return self._rate_array

