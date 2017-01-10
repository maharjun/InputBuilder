__author__ = 'Arjun'

import numpy as np
from numpy.random import mtrand as mt

from . import BaseSpikeBuilder
from ratebuilder import BaseRateBuilder
from genericbuilder.propdecorators import *

class RateBasedSpikeBuilder(BaseSpikeBuilder):
    """
        ===================
         RateBasedSpikeBuilder
        ===================
        
        Initialization
        ==============
        
        The generator is initialized by providing a dict with only two parameters 
        (specified below)

        Initialization Parameters
        *************************
        
        *rate_builder*
          The rate builder from which the spike pattern is generated.
          Note that all other information of the spike pattern, namely steps_per_ms,
          time_length, and channels are inferred from the rate generator
        
        *transform*
          This is a function that takes the rate_array returned by the rate builder,
          and returns the transformed rate array to be used for generating spikes.
          If unspecified or None, DEFAULTS TO identity transformation function np.copy()

        *start_time*
          Start time for the Spike Pattern. Default: dt

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
          This is derived from the channels of the rate generator. When set, it
          affects the channels of the rate generator. If the rate generator is
          frozen, then an exception is thrown by it.

        *rate_builder*
          The rate builder from which the spike pattern is generated.
          Note that all other information of the spike pattern, namely steps_per_ms,
          time_length, and channels are inferred from the rate generator
        
        *transform*
          This is a function that takes the rate_array returned by the rate builder,
          and returns the transformed rate array to be used for generating spikes.
          If unspecified or None, DEFAULTS TO identity transformation function np.copy()

        Other properties are documented in BaseSpikeBuilder
    """

    _shallow_copied_vars = {'_rng'}

    def __init__(self, conf_dict=None):

        self._rate_builder = BaseRateBuilder()
        self._transform = np.copy
        self._rng = mt


        super().__init__(conf_dict)

        self.rate_builder = conf_dict['rate_builder']
        self.transform = conf_dict.get('transform')
        self.rng = conf_dict.get('rng') or mt

    def _preprocess(self):
        super().with_steps_per_ms(self._rate_builder.steps_per_ms)
        super().with_time_length(self._rate_builder.time_length) 
        super().with_channels(self._rate_builder.channels)

        super()._preprocess()


    @property
    def rate_builder(self):
        """
        The Rate builder used to generate the rate for which the spikes are generated

        **GET**:
          Returns a frozen view of the rate builder. Use copy() to create a safe copy

        **SET**:
          Takes any of the following as arguments:

          1. A dict specifying the relevent properties of the rate
             generator to update.
          2. An instance of BaseRateBuilder
        """
        return self._rate_builder.frozen_view()


    # NEED TO WRITE A DICT BASED PARAMETER ASSIGNMENT FUNCTION
    @rate_builder.setter
    @requires_rebuild
    def rate_builder(self, arg):
        try:
            builder_type = arg.builder_type
        except AttributeError:
            builder_type = None

        if builder_type == 'rate':
            if not arg.is_frozen or arg.is_built:
                self._rate_builder = arg.copy()
            else:
                raise ValueError(
                    "The rate generator is a frozen, unbuilt rate generator (possible a frozen"
                    " view of an unbuilt generator), and can thus not be used to generate spikes.")
        else:
            self._rate_builder.set_properties(arg)

    @property
    def transform(self):
        return self._transform

    @transform.setter
    @requires_rebuild
    def transform(self, transform_func_):
        self._transform = transform_func_ or np.copy


    @property
    def rng(self):
        return self._rng
    
    @rng.setter
    @requires_rebuild
    def rng(self, rng_):
        self._rng = rng_


    # Overriding Base Property Setters
    @BaseSpikeBuilder.steps_per_ms.setter
    @requires_rebuild
    def steps_per_ms(self, steps_per_ms_):
        self._rate_builder.steps_per_ms = steps_per_ms_

    @BaseSpikeBuilder.time_length.setter
    @requires_rebuild
    def time_length(self, time_length_):
        self._rate_builder.time_length = time_length_

    @BaseSpikeBuilder.channels.setter
    @requires_rebuild
    def channels(self, channels_):
        self._rate_builder.channels = channels_


    def _build(self):

        super()._build()
        self._rate_builder.build()

        nchannels = self._channels.size
        array_shape = (nchannels, self._steps_length)
        poisson_distrib_spikes_from_rate = self._rng.poisson(
            lam=self._transform(self._rate_builder.rate_array)/(1000*self._steps_per_ms),
            size=array_shape)

        self._spike_rel_step_array = np.ndarray((nchannels,), dtype=object)
        self._spike_time_array = np.ndarray((nchannels,), dtype=object)
        self._spike_weight_array = np.ndarray((nchannels,), dtype=object)
        
        for i in range(nchannels):
            non_zero_spike_inds = np.argwhere(poisson_distrib_spikes_from_rate[i,:] > 0)[:, 0]
            self._spike_rel_step_array[i] = non_zero_spike_inds.astype(np.uint32)
            self._spike_weight_array[i] = poisson_distrib_spikes_from_rate[i, non_zero_spike_inds].astype(np.uint32)
            assert self._spike_rel_step_array[i].shape == self._spike_weight_array[i].shape
        
