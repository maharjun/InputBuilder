__author__ = 'Arjun'

import numpy as np

from genericbuilder.baseclass_simple import BaseGenericBuilder
from genericbuilder.propdecorators import *

from abc import abstractmethod

class BaseSpikeBuilder(BaseGenericBuilder):
    """
    Implements basic functionality for the Spike Generator Generators

    1.  Has __init__ method initializing properties common between different
        SpikeGen classes

    2.  Subscribes to the metaclass MetaSpikeGen so that all derived classes get
        the transformations done by MetaSpikeGen

    3.  Defines the properties common across different SpikeGen classes with get
        and set functions for

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

    def __init__(self):

        super().__init__()  # Initializes the core data members

        # Default Initialization of data members in case properties are not specified

        # Assignment of properties from the initialization arguments

    # ------------------------------------------------------------------------------------- #
    # CORE INTERFACE FEATURES
    # ------------------------------------------------------------------------------------- #
    # 
    # All the core interface features (abstract functions) must be implemeted in
    # the subclass

    # def _build(self):
    # def _preprocess(self):
    # def _validate(self):

    @property
    @abstractmethod
    def time_length(self):
        """
        Returns the time length of the current spike pattern in ms

        :returns: an np.float64 value representing the time length of the spike pattern

        NOTE: This is a core interface property that must be implemented (at-least the
              getter) in the subclass spike builders.
        """
        pass

    @property
    @abstractmethod
    def steps_per_ms(self):
        """
        Returns/sets the time resolution in terms of simulation steps per ms

        :returns: An np.uint32 that represents the

        NOTE: This is a core interface property that must be implemented (at-least the
              getter) in the subclass spike builders.
        """
        pass

    @property
    @abstractmethod
    def channels(self):
        """
        Returns/sets the channels for which the spikes are generated.

        :returns: a numpyp 1-D uint32 array containing the indices of the channels in
            ascending order

        NOTE: This is a core interface property that must be implemented (at-least the
              getter) in the subclass spike builders.
        """
        pass

    # @channels.setter
    # def channels(self, channels_):
    #     if channels_ is None:
    #         self._init_attr('_channels', np.zeros(0, dtype=np.uint32))
    #     else:
    #         channel_unique_array = np.array(sorted(set(channels_)), dtype=np.int32)
    #         if np.all(channel_unique_array >= 0):
    #             self._channels = np.array(channel_unique_array, dtype=np.uint32)
    #             self._channels.setflags(write=False)
    #         else:
    #             raise ValueError("'channels' should be integers >= 0")

    @property
    @requires_built
    @abstractmethod
    def spike_rel_step_array(self):
        """
        Property that returns the relative spike step array.

        :returns: an array of arrays A such that::
            
              A[i][j] = TIME STEP of the jth spike of the ith neuron relative to the
                        beginning of the spike pattern
        
        NOTE: This is a core interface property that must be implemented (only the
              getter) in the subclass spike builders.
        """
        pass

    @property
    @requires_built
    @abstractmethod
    def spike_weight_array(self):
        """
        Property that returns the spike weight array.

        :returns: an array of arrays A such that::
            
              A[i][j] = WEIGHT of the jth spike of the ith neuron
        
        NOTE: This is a core interface property that must be implemented (only the
              getter) in the subclass spike builders.
        """
        raise AttributeError(
            "The property 'spike_weight_array' is not implemented in class '{}'".format(self.__class__.__name__))

    # ------------------------------------------------------------------------------------- #
    # MIXIN INTERFACE FEATURES
    # ------------------------------------------------------------------------------------- #

    @property
    def steps_length(self):
        """
        Returns the time length of the pattern in simulation steps

        This is a mixin property that will function correctly if the core interface is
        correcty implemented
        """
        return int(self.time_length * self.steps_per_ms + 0.5)

    @requires_built
    def spike_step_array(self, start_time):
        """
        Gives the spike step array offset as though the starting time step is

            round(start_time*self.steps_per_ms)

        This is a mixin property that will function correctly if the core interface is
        correcty implemented
        """
        start_time_step = int(start_time*self.steps_per_ms + 0.5)
        ret = np.ndarray(self.spike_rel_step_array.shape, dtype=object)
        for i in range(ret.size):
            ret[i] = self.spike_rel_step_array[i] + start_time_step
            ret[i].setflags(write=False)
        ret.setflags(write=False)
        return ret

    @requires_built
    def spike_time_array(self, start_time=0):
        """
        Returns the spike time array offset as though the starting time is start_time

        :returns: an array of arrays A such that::

              A[i][j] = TIME (not Time step) of the jth spike of the ith neuron
                        assuming that start time is 'start_time'

        This is a mixin property that will function correctly if the core interface is
        correcty implemented
        """
        steps_per_ms = self.steps_per_ms
        start_time_step = int(start_time*steps_per_ms + 0.5)
        ret = np.ndarray(self.spike_rel_step_array.shape, dtype=object)
        for i in range(ret.size):
            ret[i] = (self.spike_rel_step_array[i] + start_time_step)/steps_per_ms
            ret[i].setflags(write=False)
        ret.setflags(write=False)
        return ret
