__author__ = 'Arjun'

from . import BaseSpikeBuilder
from genericbuilder.propdecorators import *
from genericbuilder.tools import *

import re

import numpy as np

class SuperimposeSpikeBuilder(BaseSpikeBuilder):
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

    def _preprocess(self):
        """
        Calculate time_length, start_time, steps_per_ms from the spike_builders.
        These functions mimic the property setting functions of the 
        """
        self._spike_builders_list = list(self._spike_builders.values())

        if self._spike_builders:
            end_time_vect = [sb.time_length + sb.start_time for sb in self._spike_builders_list]
            start_time_vect = [sb.start_time for sb in self._spike_builders.values()]

            super().with_time_length(max(end_time_vect) - min(start_time_vect))
            super().with_start_time(min(start_time_vect))
            super().with_steps_per_ms(self._spike_builders_list[0].steps_per_ms)

            channel_set_list = [set(sb.channels) for sb in self._spike_builders_list]
            super().with_channels(sorted(set.union(*channel_set_list)))

        super()._preprocess()

    # Making time_length, channels unsettable
    time_length = BaseSpikeBuilder.time_length.setter(None)
    channels    = BaseSpikeBuilder.channels.setter(None)

    # overloading the steps_per_ms setter
    @BaseSpikeBuilder.steps_per_ms.setter
    @requires_preprocessing
    @requires_rebuild
    def steps_per_ms(self, value):
        for __, spike_builder in self._spike_builders.items():
            spike_builder.steps_per_ms = value

    # Overloading the start_time setter
    @BaseSpikeBuilder.start_time.setter
    @requires_preprocessing
    @do_not_freeze
    def start_time(self, value):
        """
        If value is None, do nothing, else correct the start times of all
        constituent spike builders
        """
        if value is not None:
            curr_start_time = self.start_time
            if value > 0:
                offset = value - curr_start_time
                for spike_builder in self._spike_builders.values():
                    spike_builder.start_time = spike_builder.start_time + offset
            else:
                raise ValueError("'start_time' must be non-zero positive")


    @property
    @requires_preprocessed
    def spike_builders(self):
        new_dict = {}
        for key, val in self._spike_builders.items():
            new_dict[key] = val.frozen_view()
        return new_dict


    def update_spike_builders(self, spike_builder_list):
        """
        Update the spike builders contained. `spike_builders_list` should contain
        elements that are one of the following

        1.  A tuple ('name', spike_builder_obj)
        2.  A tuple (None, spike_builder_obj)
        3.  A spike_builder_obj

        The logic is simple:

        for each element:
            results.push_back(self.update_spike_builder(elem_spike_builder_obj, elem_name))

        where elem_name = None for cases 2, 3. The spike_builder_obj must be compatible
        with the argument restrictions enforced by update_spike_builder. 

        Importantly,

        In all cases resulting in the addition of a new spike builder, the spike_builder_obj
        must be an instance of a spike builder object (i.e. `get_builder_type()` must return
        `True`). In the case of modification of existing spike builder, the spike_builder_obj
        may be a dict (see `modify_spike_builder`)
        """

        return_spike_builder_items_list = []
        for item in spike_builder_list:
            if get_builder_type(item) == 'spike':
                # Handling input case 3
                curr_name = None
                curr_spike_builder = item
            else:
                curr_name = item[0]
                curr_spike_builder = item[1]
            return_spike_builder_items_list.append(self.update_spike_builder(curr_spike_builder, curr_name))

    def update_spike_builder(self, spike_builder_, name=None):
        """
        Update the contained spike builders with the specified entry. The logic
        is as follows
        
            if name in self.spike_builders:
                add specified spike builder
            else:
                modify existing spike builder

        The functions `add_spike_builder` and `modify_spike_builder` are used for
        this purpose. Hence the spike_builder_ object must follow restrictions
        applied by these functions
        """
        if name in self._spike_builders:
            retval = self.modify_spike_builder(spike_builder_, name)
        else:
            retval = self.add_spike_builder(spike_builder_, name)
        return retval

    def with_updated_spike_builder(self, spike_builder_, name=None):
        self.update_spike_builder(spike_builder_, name)
        return self


    @frozen_when_frozen("spike_builders")
    @requires_preprocessed
    @requires_preprocessing
    @requires_rebuild
    def add_spike_builder(self, spike_builder_, name_=None):
        """
        Add specified spike builder to set of contained spike builders.

        Parameters
        ----------

        *spike_builder_*
          This must be a spike_builder object (i.e. ``get_builder_type(spike_builder_)``
          must return 'spike' (see genericbuilder.tools))

        *name*
          This must be a string that is not already associated with any contained
          spike builder. This string can also not be of the form builder_[0-9]+ as
          this is used for default naming. If name is `None`, a default name is
          associated with it
        """
        if name_ not in self._spike_builders:
            if name_ is None:
                name_ = 'builder_{}'.format(_last_count+1)
            elif re.match(r"^builder_[0-9]+$", name_) :
                raise ValueError(
                    "The name assigned cannot be of the form builder_[0-9]+ because"
                    " this is used in the default naming scheme")

            if get_builder_type(spike_builder_) == 'spike':
                if (len(self._spike_builders) == 0
                    or spike_builder_.steps_per_ms == self._steps_per_ms):
                    if not spike_builder_.is_frozen or spike_builder_.is_built:
                        self._spike_builders[name_] = spike_builder_.copy()
                        self._last_count += 1
                    else:
                        raise ValueError(
                            "The spike builder is a frozen, unbuilt spike builder (possibly a frozen"
                            " view of an unbuilt builder), and can thus not be used to build spikes.")
                else:
                    raise ValueError("The Spike Builders must have a consistent stepsize")
            else:
                raise ValueError("The object being added is not a spike builder")
        else:
            raise ValueError("The name {} already has a spike builder associated with it".format(name_))
        

        return (name_, self._spike_builders[name_].frozen_view())

    @frozen_when_frozen("spike_builders")
    def with_added_spike_builder(self, spike_builder_, name_=None):
        self.add_spike_builder(spike_builder_, name_)
        return self

    @frozen_when_frozen("spike_builders")
    @requires_preprocessed
    @requires_preprocessing
    @requires_rebuild
    def modify_spike_builder(self, name, arg):
        """
        modifies the contained spike builder corresponding to the name `name`.

        Parameters
        ----------

        *arg*
          This can be one of the following:

          1.  A spike_builder object (i.e. ``get_builder_type(spike_builder_)``
              must return 'spike' (see genericbuilder.tools))

          2.  A dict containing properties that need be changed (using the builder
              member function set_properties()). Note that any properties that the
              object does not support setting for are simply ignored. However, if
              any supported, frozen properties are set, this does raise an exception

        *name*
          This must be a string that is associated with a contained spike builder.
        """
        if name in self._spike_builders:

            if get_builder_type(arg) == 'spikes':
                if not arg.is_frozen or arg.is_built:
                    self._rate_builders[name] = arg.copy()
                else:
                    raise ValueError(
                        "The spikes builder is a frozen, unbuilt spike builder (possibly a frozen"
                        " view of an unbuilt builder), and can thus not be used to build spikes.")
            else:
                if (len(self._spike_builders) == 1
                    or 'steps_per_ms' not in arg
                    or int(args['steps_per_ms'] + 0.5) == self._steps_per_ms):
                    self._spike_builders[name].set_properties(arg)
                else:
                    raise ValueError("The Spike Builders must have a consistent stepsize")
        else:
            raise ValueError("There is no spike builder assigned to the name {}".format(name))
        return (name, self._spike_builders[name].frozen_view())

    @frozen_when_frozen("spike_builders")
    def with_modified_spike_builder(self, name, arg):
        self.modify_spike_builder(name, arg)
        return self

    @frozen_when_frozen("spike_builders")
    def pop_spike_builder(self, name_):
        """
        Removes the spike builder corresponding to the index `index` from the list
        of indices. The index of all builders with index greater than `index` are
        reduced by 1
        """

        if name_ in self._spike_builders:
            return_builder = self._spike_builders.pop(name_)
        else:
            raise ValueError("There is no spike builder assigned to the name {}")

        return (name_, return_builder)

    @frozen_when_frozen("spike_builders")
    def with_deleted_spike_builder(self, name_):
        self.pop_spike_builder(name_)
        return self


    def _build(self):
        super()._build()

        nchannels = self._channels.size
        channel_index_map = dict(zip(self._channels, range(nchannels)))

        for builder in self._spike_builders_list:
            builder.build()

        spike_steps_appended_by_channel = [[] for __ in self._channels]
        spike_weights_appended_by_channel = [[] for __ in self._channels]
        for builder in self._spike_builders_list:
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
