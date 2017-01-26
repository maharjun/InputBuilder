__author__ = 'Arjun'

import numpy as np

from genericbuilder.baseclass import *
from genericbuilder.propdecorators import *


class BaseRateBuilder(BaseGenericBuilder):

    builder_type = 'rate' 

    def __init__(self, conf_dict=None):
        """Constructor for BaseRateGenerator
        The constructor takes one argument either named copy_object or conf_dict.

        If it is copy_object, the object must be an instance of BaseRateGenerator.
        In this case, init acts like a copy constructor.

        If it is config_dict, the argument must be dictionary subscriptable.
        The relevant keys are

        1.  steps_per_ms
        2.  time_length
        3.  channels

        When the above keys are not specified, they are initialized to their default values
        """
        super().__init__(conf_dict)
        
        if conf_dict is None:
            conf_dict = {}

        # Default initialization
        # Initialization for internal parameters
        self.time_length = None
        self.steps_per_ms = None
        self.channels = None

        # Assuming dict like object. else this will raise an exception
        self.steps_per_ms = conf_dict.get('steps_per_ms')
        self.time_length  = conf_dict.get('time_length')
        self.channels     = conf_dict.get('channels')


    def _build(self):
        super()._build()
        pass

    def _preprocess(self):
        super()._preprocess()
        self._steps_length = np.uint32(self._time_length*self._steps_per_ms + 0.5)

    def _clear(self):
        self._rate_array = np.ndarray((0,0))
        super()._clear()
        

    @property
    def steps_per_ms(self):
        """
        Get or Set the time resolution of the rate pattern by specifying an integer
        representing the number of time steps per ms

        :return:
        """
        return self._steps_per_ms

    @steps_per_ms.setter
    @requires_rebuild
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

        :GET: This function will return the time length of the built rate pattern
        i.e. the unrounded time length. not `self._steps_length/self.steps_per_ms`

        :SET: This function will round the time to the nearest time step and use
        that as the actual time length.

        :return: An np.float64 scalar
        """
        return self._time_length

    @time_length.setter
    @requires_rebuild
    def time_length(self, time_length_):
        if time_length_ is None:
            self._init_attr('_time_length', np.float64(0))
        else:
            if time_length_ >= 0:
                self._time_length  = np.float64(time_length_)
            else:
                raise ValueError("property 'time_length' must be a non-negative numeric value")


    @property
    def channels(self):
        """
        Returns read-only view of channels property. In order to make it writable,
        copy it via X.channels.copy() or np.array(X.channels)
        """
        channels_view = self._channels[:]
        channels_view.setflags(write=False)
        return channels_view

    @channels.setter
    @requires_rebuild
    def channels(self, channels_):
        # Assuming 1D iterable
        if channels_ is None:
            self._init_attr('_channels', np.ndarray(0, dtype=np.uint32))
        else:
            channel_unique_array = np.array(list(set(channels_)), dtype=np.int32)
            if np.all(channel_unique_array >= 0):
                self._channels = np.array(channel_unique_array, dtype=np.uint32)
                self._channels.setflags(write=False)
            else:
                raise ValueError("'channels' must be a vector of non-negative integers")


    @property
    @requires_preprocessed
    def time_len_in_steps(self):
        """Returns the number of time steps for which the rate pattern is built.

        :return: An np.uint32 scalar representing the number of samples
        """
        return self._steps_length

    @property
    @requires_built
    def rate_array(self):
        """
        Returns read-only view of channels property. In order to make it writable, copy it
        via X.rate_array.copy() or np.array(X.rate_array)
        """
        rate_array_view = self._rate_array[:]
        rate_array_view.setflags(write=False)
        return rate_array_view