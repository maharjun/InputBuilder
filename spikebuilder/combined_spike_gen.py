from . import BaseSpikeBuilder
from genericbuilder.propdecorators import *

import numpy as np

class CombinedSpikeBuilder(BaseSpikeBuilder):
    """
    This is a common base class that provides interface functions necessary for any spike builder that is dependent on other spike builders. In the interest of sanity, we have the following requirements
    
    1.  All the Spike Builders MUST have equal resolution i.e. steps_per_ms 
        must match

    2.  The following common properties are  are completely derived from
        the properties of the constituent spike builders.

        time_length
          max(start_time + time_length) - min(start_time). The max and min are taken
          over all the spike builders in self._final_spike_builders_list. This may or
          may not be settable depending on the derived class. If settable, it must
          adjust INDEPENDANT parameters in such a way that the above definition
          accurately reflects the new time length.

        steps_per_ms:
          The common steps_per_ms of all the spike builders. NOT SETTABLE
        
        channels: 
          The union of channels from all the spike builders. NOT SETTABLE
    
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
      as well
    """

    def __init__(self, conf_dict=None):
        self._spike_builders = {}
        self._spike_builders_name_map = {}
        self._last_count = 0
        self._final_spike_builders_list = []
        
        super().__init__(conf_dict)

        self.update_spike_builders(conf_dict.get('spike_builders') or [])

    def _preprocess(self):

        if self._final_spike_builders_list:
            spike_builders_list = self._final_spike_builders_list
            end_time_vect = [sb.time_length + sb.start_time for sb in spike_builders_list]
            start_time_vect = [sb.start_time for sb in spike_builders_list]

            spike_builders_list = list(self._spike_builders.values())
            super().with_time_length(max(end_time_vect) - min(start_time_vect))
            super().with_start_time(min(start_time_vect))
            super().with_steps_per_ms(spike_builders_list[0].steps_per_ms)
            super().with_channels(set.union(*[set(sb.channels) for sb in spike_builders_list]))
        super()._preprocess()

    def _validate(self):
        super()._validate()
        consisent_step_size = (len(set([sb.steps_per_ms for sb in self._spike_builders.values()])) <= 1)
        assert consisent_step_size, "All spike builders must have consistent stepsize"

    @property
    @requires_preprocessed
    def spike_builders(self):
        new_dict = {}
        for key, val in self._spike_builders.items():
            new_dict[key] = val.frozen_view()
        for key, val in self._spike_builders_name_map.items():
            new_dict[key] = self._spike_builders[val].frozen_view()
        return new_dict
    
    steps_per_ms = BaseSpikeBuilder.steps_per_ms.setter(None)
    channels = BaseSpikeBuilder.channels.setter(None)

    @property_setter("spike_builders")
    def update_spike_builders(self, spike_builder_list):
        """
        Update the spike builders contained. `spike_builders_list` should contain
        elements that are one of the following

        1.  A tuple (id_value, spike_builder_obj)
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
                curr_id_value = None
                curr_spike_builder = item
            else:
                curr_id_value = item[0]
                curr_spike_builder = item[1]
            return_spike_builder_items_list.append(self.update_spike_builder(curr_spike_builder, curr_id_value))
        return return_spike_builder_items_list

    @property_setter("spike_builders")
    def update_spike_builder(self, spike_builder_, id_value=None):
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

        if isinstance(id_value, str) and id_value in self._spike_builders_name_map or id_value in self._spike_builders:
            retval = self.modify_spike_builder(id_value, spike_builder_)
        else:
            retval = self.add_spike_builder(spike_builder_, id_value)
        return retval

    @property_setter("spike_builders")
    @requires_rebuild
    def add_spike_builder(self, spike_builder_, name=None):
        """
        Add specified spike builder to set of contained spike builders.

        Parameters
        ----------

        *spike_builder_*
          This must be a spike_builder object (i.e. ``get_builder_type(spike_builder_)``
          must return 'spike' (see genericbuilder.tools))

        *name*
          This must be a string. If it is already associated with a contained, it gets
          reassigned.
        """
        assert isinstance(name, str) or name is None
        if get_builder_type(spike_builder_) == 'spike':
            if not spike_builder_.is_frozen or spike_builder_.is_built:
                self._spike_builders[self._last_count + 1] = spike_builder_.copy()
                if name:
                    self._spike_builders_name_map[name] = self._last_count + 1
                self._last_count += 1
            else:
                raise ValueError(
                    "The spike builder is a frozen, unbuilt spike builder (possibly a frozen"
                    " view of an unbuilt builder), and can thus not be used to build spikes.")
        else:
            raise ValueError("The object being added is not a spike builder")

        return (self._last_count, self._spike_builders[self._last_count].frozen_view())


    @property_setter("spike_builders")
    @requires_rebuild
    def modify_spike_builder(self, id_val, arg):
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

        *id_val*
          This must either be:

          1.  A string that is associated with a contained spike builder.
          2.  A number that is a unique ID of one of the contained spike builders
        """

        if isinstance(id_val, str):
            if id_val in self._spike_builders_name_map:
                actual_uid = self._spike_builders_name_map[id_val]
            else:
                raise KeyError("There is no spike builder assigned to the name {}".format(name))
        else:
            actual_uid = int(id_val)

        if get_builder_type(arg) == 'spike':
            if not arg.is_frozen or arg.is_built:
                self._spike_builders[actual_uid] = arg.copy()
            else:
                raise ValueError(
                    "The spikes builder is a frozen, unbuilt spike builder (possibly a frozen"
                    " view of an unbuilt builder), and can thus not be used to build spikes.")
        else:
            self._spike_builders[actual_uid].set_properties(arg)


        return (actual_uid, self._spike_builders[actual_uid].frozen_view())


    @property_setter("spike_builders")
    def pop_spike_builder(self, id_val):
        """
        Removes the spike builder corresponding to the index `index` from the list
        of indices. all corresponding name maps are removed
        """

        if isinstance(str, id_val):
            if id_val in self._spike_builders_name_map:
                actual_uid = self._spike_builders_name_map[id_val]
            else:
                raise KeyError("There is no spike builder assigned to the name {}".format(name))
        else:
            actual_uid = id_val
        
        name_map_list = list(self._spike_builders_name_map.items())
        name_map_list = [x for x in name_map_list if x[1] == actual_uid]
        for elem in name_map_list:
            self._spike_builders_name_map.pop(x)
        return_builder = self._spike_builders.pop(actual_uid)

        return (actual_uid, return_builder)

    @property
    def buildercount(self):
        return self._last_count

    def _build(self):
        super()._build()

        nchannels = self._channels.size
        channel_index_map = dict(zip(self._channels, range(nchannels)))

        for builder in self._final_spike_builders_list:
            builder.build()

        spike_steps_appended_by_channel = [[] for __ in self._channels]
        spike_weights_appended_by_channel = [[] for __ in self._channels]
        for builder in self._final_spike_builders_list:
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