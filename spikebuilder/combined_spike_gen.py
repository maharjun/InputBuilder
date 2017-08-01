from . import BaseSpikeBuilder
from genericbuilder.propdecorators import requires_built
from genericbuilder.tools import get_builder_type

import numpy as np


class CombinedSpikeBuilder(BaseSpikeBuilder):
    """
    This is the class that implements mechanisms for generating combined output
    from one or more spike builders.

    Initialization
    ==============

    :param spike_builder_list: This is the list of the constituent spike builders.
        Each of the spike builders must have the same resolution.

    :param repeat_instance_list: This is a list of repeat instances that represents how
        the above mentioned spike builders are combined.

        The repeat instances are 2 or 3-tuples with the first value being the
        starting time (>=0), and the second value being the index of the spike
        builder to use, and the third value if specified must be the time length to
        stretch the spike builder.

    :param time_length: This must either be a float >= 0 or the string 'auto'. If
        'auto' then the time length is inferred from the time lengths of the spike
        builders according to their placement as described in the
        `repeat_instance_list`.

        If the time length is specified (as a float i.e.), Then it is necessary
        that this float is greater than the time length as inferred in the above
        case. In this case, the spike builder will create an 'empty' spike train
        (i.e. by generating no spikes) for the excess duration

    Properties
    ==========

    Settable
    --------

    *time_length*: This is the time length as specified in the initialization
      function. When getting the value, it is either automatically calculated (if
      previously set to 'auto') or the previously set value is returned

    *spike_builders*: This is the spike builders list as specified in the
      initialization function. when getting, a tuple of IMMUTABLE spike builders is
      returned

    *repeat_instances*: This is the repeat_instances_list as specified in the
      initialization function.  When getting it, a tuple of repeat_instances is
      returned

    Non-Settable
    ------------

    *steps_per_ms*:
      resolution ins steps_per_ms

    *channels*:
      The array of channels

    *spike_rel_step_array*:
      See BaseSpikeBuilder

    *spike_weight_array*:
      See BaseSpikeBuilder
    """

    def __init__(self, spike_builder_list=[], repeat_instances_list=[], time_length='auto'):
        # self._final_spike_builders_list = [] [Must be set in preprocess of derived class]

        # Filtering Input Dict

        # Super Initialization
        super().__init__()  # only purpose is to run BaseGenericBuilder init

        # Initialization of properties
        self.spike_builders = spike_builder_list
        self.repeat_instances = repeat_instances_list
        self.time_length = time_length

    def _get_time_length_from_ri(self, ri):
        if ri[2] is None:
            return self._spike_builders[ri[1]].time_length
        else:
            return ri[2]

    def _preprocess(self):
        if self._spike_builders:
            common_channels = set.union(*[set(sb.channels) for sb in self._spike_builders])
            self._channels = np.array(sorted(common_channels), dtype=np.uint32)
        else:
            self._channels = np.array(0, dtype=np.uint32)

        if self._time_length_is_derived:
            if self._spike_builders:
                self._time_length = max(ri[0] + self._get_time_length_from_ri(ri)
                                        for ri in self._repeat_instances)
            else:
                self._time_length = 0

    def _validate(self):
        if not self._time_length_is_derived:
            max_time_length = max(ri[0] + self._get_time_length_from_ri(ri)
                                  for ri in self._repeat_instances)
            assert max_time_length <= self._time_length, \
                "The Constituent Spike Builders are not within time bounds"

        consisent_step_size = (len(set(sb.steps_per_ms for sb in self._spike_builders)) <= 1)
        assert consisent_step_size, "All spike builders must have consistent stepsize"

    @property
    def steps_per_ms(self):
        if len(self._spike_builders):
            return self._spike_builders[0].steps_per_ms
        else:
            return 1

    @property
    def channels(self):
        return self._channels

    @property
    def time_length(self):
        return self._time_length

    @time_length.setter
    def time_length(self, time_length_):
        if time_length_ == 'auto':
            self._time_length_is_derived = True
        else:
            if time_length_ >= 0:
                self._time_length = np.float64(time_length_)
                self._time_length_is_derived = False
            else:
                raise ValueError("'time_length' must be a non-negative number")

    @property
    def spike_builders(self):
        return self._spike_builders

    @spike_builders.setter
    def spike_builders(self, spike_builder_list):
        assert all(get_builder_type(sb) == 'spike' for sb in spike_builder_list), \
            "'spike_builder_list' must contain objects inheriting from BaseSpikeBuilder"
        self._spike_builders = tuple(sb.copy().set_immutable() for sb in spike_builder_list)

    @property
    def buildercount(self):
        return len(self._spike_builders)

    @property
    def repeat_instances(self):
        return self._repeat_instances

    @repeat_instances.setter
    def repeat_instances(self, repeat_instances_list):
        assert all(len(ri) in {2, 3} and ri[0] >= 0 and 0 <= ri[1] < len(self._spike_builders)
                   for ri in repeat_instances_list), \
            ("The repeat instances must be 2 or 3-tuples with the first value being the"
             " starting time (>=0), and the second value being the index of the spike"
             " builder to use, and the third value if specified must be the time length"
             " to stretch the spike builder.")
        repeat_instances_list = [(ri[0], ri[1], None if len(ri) == 2 else ri[2])
                                 for ri in repeat_instances_list]
        self._repeat_instances = tuple(sorted(repeat_instances_list, key=lambda x: x[0:2]))

    def _build(self):

        nchannels = self.channels.size
        channel_index_map = dict(zip(self.channels, range(nchannels)))

        # Performing Builds for each repeat instance. Note the builds are performed
        # sequentially. this means that if there is any state that is maintained after
        # each run, that state is incremented for each build of a particular generator
        # The self._spike_builders array is updated to reflect the new post-built
        # generators
        spike_builders_list = list(self._spike_builders)
        built_builders = []  # create one built copy of the builder for each repeat index
        for start, index, stretch in self._repeat_instances:
            current_sb = spike_builders_list[index].copy_mutable()
            if stretch is not None:
                current_sb.time_length = stretch
            spike_builders_list[index] = current_sb.build().set_immutable()
            built_builders.append(current_sb)
        self._spike_builders = tuple(spike_builders_list)

        # Join the spike-steps and weights channel-wise across builders
        spike_steps_clubbed = [[] for __ in self._channels]
        spike_weights_clubbed = [[] for __ in self._channels]
        for i, builder in enumerate(built_builders):
            for (channel,
                 channel_spike_step_array,
                 channel_spike_weight_array) in zip(builder.channels,
                                                    builder.spike_step_array(start_time=self._repeat_instances[i][0]),
                                                    builder.spike_weight_array):

                spike_steps_clubbed[channel_index_map[channel]].append(channel_spike_step_array)
                spike_weights_clubbed[channel_index_map[channel]].append(channel_spike_weight_array)

        # Concatenating arrays for each channel
        spike_steps_joined = [np.concatenate(x) for x in spike_steps_clubbed]
        spike_weights_joined = [np.concatenate(x) for x in spike_weights_clubbed]

        # Grouping together the weights of the spikes that happen in a single time step
        # and sorting the time step values
        spike_steps_unique = [np.lib.arraysetops.unique(x) for x in spike_steps_joined]
        spike_weights_joined = [np.bincount(step_inds, weights=weights)[step_inds_unique]
                                for step_inds, step_inds_unique, weights in zip(spike_steps_joined,
                                                                                spike_steps_unique,
                                                                                spike_weights_joined)]
        spike_steps_joined = spike_steps_unique

        # Convert explicitly to read-only list-of-lists to avoid collapsing into 2-D lists
        self._spike_rel_step_array = np.ndarray(len(spike_steps_joined), dtype=object)
        self._spike_weight_array = np.ndarray(len(spike_weights_joined), dtype=object)

        for i, (spike_rel_step, spike_weight) in enumerate(zip(spike_steps_joined, spike_weights_joined)):
            self._spike_rel_step_array[i] = spike_rel_step
            self._spike_weight_array[i] = spike_weight
            self._spike_rel_step_array[i].setflags(write=False)
            self._spike_weight_array[i].setflags(write=False)

        self._spike_rel_step_array.setflags(write=False)
        self._spike_weight_array.setflags(write=False)

    @property
    @requires_built
    def spike_rel_step_array(self):
        return self._spike_rel_step_array

    @property
    @requires_built
    def spike_weight_array(self):
        return self._spike_weight_array
