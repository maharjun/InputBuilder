from . import BaseSpikeBuilder
from genericbuilder.propdecorators import prop_setter, prop_getter, requires_built
from genericbuilder.tools import get_builder_type, ImmutableDict
from timeit import default_timer as timer

import numpy as np

class CombinedSpikeBuilder(BaseSpikeBuilder):
    """
    This is a common base class that provides interface functions necessary for any
    spike builder that is dependent on other spike builders. In the interest of sanity,
    we have the following requirements
    
    1.  All the Spike Builders MUST have equal resolution i.e. steps_per_ms must match

    2.  The interaction of the common properties with the properties of the constituent
        spike biulders are as follows.

        time_length
          This is implementation specific. It may be a completely independent property,
          in which case, no work need be done. If it is a dependant property, then the
          calculation and assignment of the same should be done in the _preprocess
          function prior to calling the _preprocess funtion of the CombinedSpikeBuilder
          class. Naturally, the settability must be decided by the subclass

        steps_per_ms:
          The common steps_per_ms of all the spike builders. NOT SETTABLE
        
        channels: 
          The union of channels from all the spike builders. NOT SETTABLE

        start_time:
          This is as in BaseSpikeBuilder. i.e. it is an independant variable.

          *NOTE*:
            The start time of each of the time patterns in self._final_spike_builders_list
            is treated RELATIVE to the start time of the combined pattern when building
            spikes.
    
    It has the following structure:

    1.  A self._spike_builders variable that gets edited based on the spike builders
        added and removed via the functions defined below

    2.  A self._spike_builders_name_map variable that contains any name mappings
        defined for the generators in self._spike_builders

    3.  A self._final_spike_builders_list that is populated with the actual spike builders
        whose outputs are linearly combined. This is a derived variable and must
        be set in the _preprocess function of the inheriting class before the
        _preprocess of this class is called.

        The derivation of self._final_spike_builders_list from self._spike_builders
        is the defining attribute of the derived class. Other defining attributes
        are additional validations, variables, setting of time_length and start_time
        
    Initialization
    ==============

    The only parameters parsed in the input dictionary are:

    *spike_builders*
      This represents the spike builders in this spike generator

      This is a python iterable whose elements are one of the following
        1. A tuple ('name', spike_builder_obj)
        2. A tuple (None, spike_builder_obj)
        3. A spike_builder_obj

      Every spike builder added is assigned a unique identifying number (incremented
      every add). If Name is specified, then the said builder may be addressed by name
      as well.

    *start_time*
      This assigns the start time of the spikebuilder.

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

    def _preprocess(self):
        if self._spike_builders:
            self._steps_per_ms = self._spike_builders[0].steps_per_ms
            common_channels = set.union(*[set(sb.channels) for sb in self._spike_builders])
            self._channels = np.array(sorted(common_channels), dtype=np.uint32)
        else:
            self._steps_per_ms = 1
            self._channels = np.array(0, dtype=np.uint32)

        if self._time_length_is_derived:
            if self._spike_builders:
                self._time_length = max(start + self._spike_builders[ind].steps_length/self._steps_per_ms
                                        for start, ind in self._repeat_instances)
            else:
                self._time_length = 0

    def _validate(self):
        if not self._time_length_is_derived:
            max_time_length = max(start + self._spike_builders[ind].steps_length/self._steps_per_ms
                                  for start, ind in self._repeat_instances)
            assert max_time_length <= self._time_length, \
                   "The Constituent Spike Builders are not within time bounds"

        consisent_step_size = (len(set(sb.steps_per_ms for sb in self._spike_builders)) <= 1)
        assert consisent_step_size, "All spike builders must have consistent stepsize"

    @property
    def steps_per_ms(self):
        return self._steps_per_ms

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
        return self._repeat_instances_sorted

    @repeat_instances.setter
    def repeat_instances(self, repeat_instances_list):
        assert all(len(ri) == 2 and ri[0] >= 0 and 0 <= ri[1] < len(self._spike_builders)
                   for ri in repeat_instances_list), \
            ("The repeat instances must be 2-tuples with the first instance being the"
             " starting time (>=0), and the second instance being the index of the spike"
             " builder to use")
        self._repeat_instances = tuple(sorted(repeat_instances_list))

    def _build(self):

        nchannels = self._channels.size
        channel_index_map = dict(zip(self.channels, range(nchannels)))

        # Performing Builds for each repeat instance. Note the builds are performed
        # sequentially. this means that if there is any state that is maintained after
        # each run, that state is incremented for each build of a particular generator
        # The self._spike_builders array is updated to reflect the new post-built
        # generators
        time_build_start = timer()
        spike_builders_list = list(self._spike_builders)
        built_builders = []  # create one built copy of the builder for each repeat index
        for start, index in self._repeat_instances:
            spike_builders_list[index] = spike_builders_list[index].build_copy()
            built_builders.append(spike_builders_list[index])        
        self._spike_builders = tuple(spike_builders_list)
        time_build_end = timer()

        time_join_start = timer()

        # Join the spikesteps and weights channel-wise accross builders
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
        spike_weights_joined = [np.bincount(step_inds, weights=weights)[np.lib.arraysetops.unique(step_inds)]
                                  for step_inds, step_inds_unique, weights in zip(spike_steps_joined,
                                                                                  spike_steps_unique,
                                                                                  spike_weights_joined)]
        spike_steps_joined = spike_steps_unique

        self._spike_rel_step_array = np.array(spike_steps_joined, dtype=object)
        self._spike_weight_array   = np.array(spike_weights_joined, dtype=object)
        time_join_end = timer()

        self._spike_rel_step_array.setflags(write=False)
        self._spike_weight_array.setflags(write=False)

        print("Building Time: {} Joining Time: {}".format(time_build_end - time_build_start,
                                                          time_join_end - time_join_start))

    @property
    @requires_built
    def spike_rel_step_array(self):
        return self._spike_rel_step_array

    @property
    @requires_built
    def spike_weight_array(self):
        return self._spike_weight_array
