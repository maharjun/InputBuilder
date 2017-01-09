__author__ = 'Arjun'

import numpy as np

from genericbuilder.baseclass import BaseGenericBuilder
from genericbuilder.propdecorators import *


class BaseSpikeBuilder(BaseGenericBuilder):
    """
    Implements basic functionality for the Spike Generator Generators

    1.  Has __init__ method initializing properties common between different
        SpikeGen classes

    2.  Subscribes to the metaclass MetaSpikeGen so that all derived classes
        get the transformations done by MetaSpikeGen

    3.  Defines the properties common across different SpikeGen classes with
        get and set functions for 

    4.  All properties steps_per_ms, time_length, and channels have their get
        functions decorated by requires_preprocessed. This is because in
        SpikeBuilder, they will be derived properties of the rate generators/
        contained spike generators in many cases.

    5.  The build_ function should set the following members

            self._spike_rel_step_array
            self._spike_weight_array

        The following properties can be used to access these members

            spike_time_array
            spike_step_array
            spike_rel_step_array
            spike_weight_array
    """

    builder_type = 'spike'

    def __init__(self, conf_dict=None):

        super().__init__(conf_dict)

        self._time_length = 0. 
        self._steps_length = 0
        self._steps_per_ms = 1
        self._channels = np.ndarray((0,), dtype=np.float64())

        self._spike_time_array = np.ndarray((0,), dtype=object)
        self._spike_rel_step_array = np.ndarray((0,), dtype=object)
        self._spike_weight_array = np.ndarray((0,), dtype=object)

        self._start_time = None
        self._start_time_step = 1

        self._default_props = {
            'time_length': 0,
            'channels': [],
            'steps_per_ms': 1,
            'start_time': None
        }

        updated_props = self._default_props
        updated_props.update(conf_dict)

        self.time_length = updated_props['time_length']
        self.steps_per_ms = updated_props['steps_per_ms']
        self.channels = updated_props['channels']
        self.start_time = updated_props['start_time']

    def _build(self):
        super()._build()
        
    def _preprocess(self):
        """Performs the follwing preprocessing on the class:

        1.  Calculates the value of _steps_length
        2.  Initializes _start_time to be equal to dt (since 0 results in NEST Error (Why?))
        3.  Recalculates self._spike_time_array (relevant when start_time changes)
        """
        super()._preprocess()
        self._steps_length = np.uint32(self._time_length * self._steps_per_ms + 0.5)
        if self._start_time is not None:
            self._start_time_step = max(np.uint32(1),
                                        np.uint32(self._start_time*self._steps_per_ms + 0.5))
        else:
            self._start_time_step = np.uint32(1)


    @property
    @requires_preprocessed
    def time_length(self):
        return self._time_length

    @time_length.setter
    @requires_rebuild
    def time_length(self, time_length_):
        if time_length_ >= 0:
            self._time_length = np.float64(time_length_)
        else:
            raise ValueError("'time_length' must be non-negative")


    @property
    @requires_preprocessed
    def steps_per_ms(self):
        return self._steps_per_ms

    @steps_per_ms.setter
    @requires_rebuild
    def steps_per_ms(self, steps_per_ms_):
        if steps_per_ms_ >= 1:
            self._steps_per_ms = np.uint32(steps_per_ms_)
        else:
            raise ValueError("'steps_per_ms' must be >= 1")


    @property
    @requires_preprocessed
    def channels(self):
        channels_ro = self._channels[:]
        channels_ro.setflags(write=False)
        return channels_ro

    @channels.setter
    @requires_rebuild
    def channels(self, channels_):
        channels_np = np.array(channels_)
        if np.all(channels_np >= 1):
            self._channels = channels_np.astype(np.uint32)
        else:
            raise ValueError("'channels' should be integers >= 1")


    @property
    @requires_preprocessed
    def start_time(self):
        return self._start_time_step/self._steps_per_ms

    @start_time.setter
    @do_not_freeze
    def start_time(self, start_time_):
        if start_time_ is not None:
            if start_time_ > 0:
                self._start_time = np.float64(start_time_)
            else:
                raise ValueError("'start_time' must be non-zero positive")
        else:
            self._start_time = None

    @property
    @requires_preprocessed
    def steps_length(self):
        return self._steps_length

    @property
    @requires_built
    def spike_rel_step_array(self):
        ret = np.ndarray(self._spike_rel_step_array.shape, dtype=object)
        for i in range(ret.size):
            ret[i] = self._spike_rel_step_array[i][:]
            ret[i].setflags(write=False)
        ret.setflags(write=False)
        return ret

    @property
    @requires_built
    def spike_weight_array(self):
        ret = np.ndarray(self._spike_weight_array.shape, dtype=object)
        for i in range(ret.size):
            ret[i] = self._spike_weight_array[i][:]
            ret[i].setflags(write=False)
        ret.setflags(write=False)
        return ret

    @property
    @cached('spike_step_array')
    @requires_built
    def spike_step_array(self):
        ret = np.ndarray(self._spike_rel_step_array.shape, dtype=object)
        for i in range(ret.size):
            ret[i] = self._spike_rel_step_array[i] + self._start_time_step
            ret[i].setflags(write=False)
        ret.setflags(write=False)
        return ret

    @property
    @cached('spike_time_array')
    @requires_built
    def spike_time_array(self):
        ret = np.ndarray(self._spike_time_array.shape, dtype=object)
        for i in range(ret.size):
            ret[i] = (self._spike_rel_step_array[i] + self._start_time_step)/self._steps_per_ms
            ret[i].setflags(write=False)
        ret.setflags(write=False)
        return ret