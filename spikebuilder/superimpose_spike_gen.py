__author__ = 'Arjun'

from . import CombinedSpikeBuilder
from genericbuilder.propdecorators import *
from genericbuilder.tools import *

import numpy as np

class SuperimposeSpikeBuilder(CombinedSpikeBuilder):
    """
    This spike builder superimposes multiple spike trains on each other to create
    a resultant spike train. in ordr to make sense we have the following scheme.
    
    1.  All the Spike Builders MUST have equal resolution i.e. steps_per_ms 
        must match

    2.  The common properties are calculated from the input spike trains as follows

        time_length
          max(start_time + time_length) - min(start_time). This is NOT settable
          (being a purely derived property)

        steps_per_ms:
          The common steps_per_ms of all the spike builders. When set, it affects
          the steps_per_ms of all rate builders
        
        channels: 
          The union of channels from all the spike builders. Note that this is
          NOT settable (purely dervived property)

        start_time:
          Inferred from min(start_time). When set, it offsets the start_times of
          all the contained builders by the required amount maintaining the same
          relative distance

    Initialization
    ==============

    The only parameters parsed in the input dictionary are:

    *spike_builders*
      This represents the spike builders in this spike generator

      This is a python iterable whose elements are one of the following
        1. A tuple ('name', spike_builder_obj)
        2. A tuple (None, spike_builder_obj)
        3. A spike_builder_obj

    *start_time*
      This specifies the start_time property
    """

    def __init__(self, conf_dict=None):
        self._spike_builders = {}
        self._last_count = 0
        
        super().__init__(conf_dict)

        self.update_spike_builders(conf_dict.get('spike_builders') or [])
        self.start_time = conf_dict.get('start_time')

    def _validate(self):
        consisent_step_size = (len(set([sb.steps_per_ms for sb in self._spike_builders.values()])) <= 1)
        assert consisent_step_size, "All spike builders must have consistent stepsize"

    def _preprocess(self):
        """
        Calculate time_length, start_time, from the spike_builders.
        These functions mimic the property setting functions of the 
        """

        if self._spike_builders:
            spike_builders_list = list(self._spike_builders.values())
            end_time_vect = [sb.time_length + sb.start_time for sb in spike_builders_list]
            start_time_vect = [sb.start_time for sb in spike_builders_list]

            super().with_time_length(max(end_time_vect) - min(start_time_vect))
            super().with_start_time(min(start_time_vect))
            
        super()._preprocess()

    # Making time_length unsettable
    time_length = CombinedSpikeBuilder.time_length.setter(None)

    # Overloading the start_time setter
    @CombinedSpikeBuilder.start_time.setter
    @do_not_freeze
    def start_time(self, value):
        """
        If value is None, do nothing, else correct the start times of all
        constituent spike builders
        """
        if value is not None:
            curr_start_time = min([sb.start_time for sb in self._spike_builders.values()])
            if value > 0:
                offset = value - curr_start_time
                for spike_builder in self._spike_builders.values():
                    spike_builder.start_time = spike_builder.start_time + offset
            else:
                raise ValueError("'start_time' must be non-zero positive")


    def _build(self):
        super()._build()

        nchannels = self._channels.size
        channel_index_map = dict(zip(self._channels, range(nchannels)))

        for builder in self._spike_builders.values():
            builder.build()

        spike_steps_appended_by_channel = [[] for __ in self._channels]
        spike_weights_appended_by_channel = [[] for __ in self._channels]
        for builder in self._spike_builders.values():
            for (channel,
                 channel_spike_step_array,
                 channel_spike_weight_array) in zip(builder.channels,
                                                    builder.spike_step_array,
                                                    builder.spike_weight_array):

                spike_steps_appended_by_channel[channel_index_map[channel]].append(channel_spike_step_array)
                spike_weights_appended_by_channel[channel_index_map[channel]].append(channel_spike_weight_array)

        for i in range(nchannels):
            spike_bincount_vector = np.zeros((self._steps_length,), dtype=np.uint32)
            for j in range(len(spike_steps_appended_by_channel[i])):
                curr_spike_list = spike_steps_appended_by_channel[i][j]
                curr_weight_list = spike_weights_appended_by_channel[i][j]
                spike_bincount_vector[curr_spike_list - self._start_time_step] += curr_weight_list

            # Note that spike_steps_app... now represents the relative steps thanks to 
            # curr_spike_list - self._start_time_step above and np.argwhere
            non_zero_spike_indices = np.argwhere(spike_bincount_vector)[:, 0]
            spike_steps_appended_by_channel[i] = non_zero_spike_indices
            spike_weights_appended_by_channel[i] = spike_bincount_vector[non_zero_spike_indices]

        self._spike_rel_step_array = np.array(spike_steps_appended_by_channel)
        self._spike_weight_array   = np.array(spike_weights_appended_by_channel, dtype=object)
