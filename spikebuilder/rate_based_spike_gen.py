__author__ = 'Arjun'

import numpy as np
from numpy.random import mtrand

from . import BaseSpikeBuilder
from ratebuilder import BaseRateBuilder
from genericbuilder.tools import get_builder_type
from genericbuilder.propdecorators import requires_built

mtgen = mtrand.binomial.__self__

class RateBasedSpikeBuilder(BaseSpikeBuilder):
    """
        ===================
         RateBasedSpikeBuilder
        ===================
        
        Initialization
        ==============
        
        The builder is initialized by providing a dict with only two parameters 
        (specified below)

        Initialization Parameters
        *************************
        
        *rate_builder*
          The rate builder from which the spike pattern is generated. Note that all
          other information of the spike pattern, namely steps_per_ms, time_length, and
          channels are inferred from the rate generator
        
        *transform*
          This is a function that takes the rate_array returned by the rate builder,
          and returns the transformed rate array to be used for generating spikes. If
          unspecified or None, DEFAULTS TO identity transformation function np.copy()

        *start_time*
          Start time for the Spike Pattern. Default: dt
        
        *rng*
          The RandomSate object that is used for random generation. Defaults to numpy
          default

        Properties
        ==========

        *time_length*
          This is derived from the time length of the rate generator. When set, it
          affects the time length of the rate generator. If the rate generator is
          frozen, then an exception is thrown by it.

        *steps_per_ms*
          This is derived from the steps_per_ms of the rate generator. When set, it
          affects the steps_per_ms of the rate generator. If the rate generator is
          frozen, then an exception is thrown by it.

        *channels*
          This is derived from the channels of the rate generator. When set, it affects
          the channels of the rate generator. If the rate generator is frozen, then an
          exception is thrown by it.

        *rate_builder*
          The rate builder from which the spike pattern is generated. Note that all
          other information of the spike pattern, namely steps_per_ms, time_length, and
          channels are inferred from the rate generator
        
        *transform*
          This is a function that takes the rate_array returned by the rate builder,
          and returns the transformed rate array to be used for generating spikes. If
          unspecified or None, DEFAULTS TO identity transformation function np.copy()

        Other properties are documented in BaseSpikeBuilder
    """


    def __init__(self, rate_builder, transform=np.copy, rng=mtgen):

        # default init of super and current class
        super().__init__()  # only purpose is to run BaseGenericBuilder init

        # Initializing Data members in case corresponding properties are not specified
        
        # Assigning properties from arguments
        self.rate_builder = rate_builder
        self.transform = transform
        self.rng = rng        

    def _preprocess(self):
        # No preprocessing required for this class
        pass

    def _validate(self):
        pass

    @property
    def rate_builder(self):
        """
        The Rate builder used to generate the rate for which the spikes are generated

        **GET**:
          Returns a frozen view of the rate builder. Use copy() to create a safe copy

        **SET**:
          Takes any of the following as arguments:

          1. An instance of BaseRateBuilder
        """
        return self._rate_builder  # Note that _rate_builder is immutable


    @rate_builder.setter
    def rate_builder(self, rate_builder_):
        if get_builder_type(rate_builder_) == 'rate':
            self._rate_builder = rate_builder_.copy().set_immutable()
        else:
            raise TypeError("'rate_builder_' must be an instance of BaseRateBuilder")

    @property
    def transform(self):
        return self._transform

    @transform.setter
    def transform(self, transform_func_):
        self._transform = transform_func_


    @property
    def rng(self):
        return self._rng
    
    @rng.setter
    def rng(self, rng_):
        self._rng = rng_


    # Overriding Base Property Setters
    @property
    def steps_per_ms(self):
        return self._rate_builder.steps_per_ms

    @steps_per_ms.setter
    def steps_per_ms(self, steps_per_ms_):
        self._rate_builder = self._rate_builder.copy_mutable()
        self._rate_builder.steps_per_ms = steps_per_ms_
        self._rate_builder = self._rate_builder.set_immutable()


    @property
    def time_length(self):
        return self._rate_builder.time_length

    @time_length.setter
    def time_length(self, time_length_):
        self._rate_builder = self._rate_builder.copy_mutable()
        self._rate_builder.time_length = time_length_
        self._rate_builder = self._rate_builder.set_immutable()


    @property
    def channels(self):
        return self._rate_builder.channels

    @channels.setter
    def channels(self, channels_):
        self._rate_builder = self._rate_builder.copy_mutable()
        self._rate_builder.channels = channels_
        self._rate_builder = self._rate_builder.set_immutable()

    @property
    @requires_built
    def spike_rel_step_array(self):
        """
        Property that returns the relative spike step array.

        :returns: an array of arrays A such that::
            
              A[i][j] = TIME STEP of the jth spike of the ith neuron relative to the
                        beginning of the spike pattern
        """
        return self._spike_rel_step_array

    @property
    @requires_built
    def spike_weight_array(self):
        """
        Property that returns the spike weight array.

        :returns: an array of arrays A such that::
            
              A[i][j] = WEIGHT of the jth spike of the ith neuron
        """
        return self._spike_weight_array

    def _build(self):

        super()._build()
        self._rate_builder = self._rate_builder.build_copy()

        nchannels = self.channels.size
        array_shape = (nchannels, self.steps_length)
        poisson_distrib_spikes_from_rate = self._rng.poisson(
            lam=self._transform(self._rate_builder.rate_array)/(1000*self.steps_per_ms),
            size=array_shape)

        self._spike_rel_step_array = np.ndarray((nchannels,), dtype=object)
        self._spike_weight_array = np.ndarray((nchannels,), dtype=object)
        
        for i in range(nchannels):
            non_zero_spike_inds = np.argwhere(poisson_distrib_spikes_from_rate[i,:] > 0)[:, 0]
            self._spike_rel_step_array[i] = non_zero_spike_inds.astype(np.uint32)
            self._spike_weight_array[i] = poisson_distrib_spikes_from_rate[i, non_zero_spike_inds].astype(np.uint32)
            assert self._spike_rel_step_array[i].shape == self._spike_weight_array[i].shape

        self._spike_rel_step_array.setflags(write=False)
        self._spike_weight_array.setflags(write=False)
        
