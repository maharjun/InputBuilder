from . import CombinedSpikeBuilder
from genericbuilder.propdecorators import *
from collections import namedtuple
import numpy as np

class RepeaterSpikeBuilder(CombinedSpikeBuilder):
    """
    ======================
     RepeaterSpikeBuilder
    ======================

    This spike builder enables the creations of patterns that are copies of a fixed
    set of base patterns.

    Relevant Properties and Initialization
    ======================================

    The base patterns are given as input following the syntax in CombinedSpikeBuilder.
    The copies are presented to the class as tuples of (start_time, spike_builder_id).
    Where spike_builder_id is either the associated name, or the id number of the
    spike builder that you want placed at start_time. the start_time is specified in
    ms relative to the start of the RepeaterSpikeBuilder pattern. each such copy is
    termed as **repeat instance**

    Init Parameters
    ---------------

    In the init function, we have the following relevant parameters.

    spike_builders:

      This is he same as documented in CombinedSpikeBuilder

    start_time:

      The start time of the RepeaterSpikeBuilder pattern in ms

    Editing Repeat instances
    ------------------------

    Repeat instances can be added or deleted using the `add_repeat_instance` and
    `pop_repeat_instance` functions. See these function docs for details
    """
    def __init__(self, conf_dict):
        # Default Init done directly here because of no property setter that can
        # take None argument effectively. will deal with this later if necessary

        super().__init__({})
        self._repeat_instances_set = frozenset()
        self._repeat_instances_sorted = []  # derived from 
        self.time_length = None

        conf_dict_base = {key:conf_dict.get(key) for key in ['spike_builders', 'start_time']}

        super().__init__(conf_dict_base)
        self.time_length = conf_dict.get('time_length')


    def _validate(self):
        super()._validate()
        start_end_time_list = [(sb_st, sb_st + self._spike_builders[sb_uid].time_length)
                               for sb_st, sb_uid in self._repeat_instances_set]
        start_end_time_list = sorted(start_end_time_list)
        is_disjoint_list = [start_end_time_list[i+1][0] >= start_end_time_list[i][1]
                            for i in range(len(start_end_time_list) - 1)]
        is_disjoint = all(is_disjoint_list)
        assert is_disjoint, "All the constituent spike builders must correspond to disjoint intervals"

        if self._is_time_length_assigned:
            is_within_time_bounds = max(sb_st + self._spike_builders[sb_uid].time_length
                                        for sb_st, sb_uid in self._repeat_instances_set) <= self._time_length
            assert is_within_time_bounds, "The specified repeats are not within the time bounds"

    def _preprocess(self):
        
        # Adding Actual Spike Builders 
        self._repeat_instances_sorted = sorted(self._repeat_instances_set)
        actual_spike_builders_list = [self._spike_builders[sb_uid].copy_mutable()
                                                                  .with_start_time(sb_st)
                                                                  .copy_immutable()
                                      for sb_st, sb_uid in self._repeat_instances_sorted]
        
        # Adding Filler Spike Builders if present
        filler_spike_builders_list = []
        if 'filler' in self._spike_builders_name_map:
            filler_uid = self._spike_builders_name_map['filler']
            all_start_end_times = [(sb_st, self._spike_builders[sb_uid].time_length + sb_st)
                                   for sb_st, sb_uid in self._repeat_instances_sorted]
            filler_start_end_times = [(all_start_end_times[i][1], all_start_end_times[i+1][0])
                                      for i in range (len(self._repeat_instances_sorted)-1)]

            filler_sb = self._spike_builders[filler_uid]
            filler_spike_builders_list = [
                filler_sb.copy_mutable().with_start_time(fill_beg)
                                        .with_time_length(fill_end - fill_beg)
                                        .copy_immutable()
                for fill_beg, fill_end in filler_start_end_times]

        self._final_spike_builders_list = actual_spike_builders_list + filler_spike_builders_list
        
        # If the time length is not manually assigned, infer it from the spike builders
        if not self._is_time_length_assigned:
            time_length = max(sb_st + self._spike_builders[sb_uid].time_length
                              for sb_st, sb_uid in self._repeat_instances_set)
            super().with_time_length(time_length)
        super()._preprocess()

    def _build(self):
        super()._build()

    
    @CombinedSpikeBuilder.time_length.setter
    def time_length(self, time_length_):
        if time_length_ is None:
            super().with_time_length(None)
            self._is_time_length_assigned = False
        elif time_length_ == 0:
            self._is_time_length_assigned = False
        else:
            super().with_time_length(time_length_)
            self._is_time_length_assigned = True

    @property_setter('repeat_instances')
    def add_repeat_instance(self, start_time, sb_id_val):
        """
        This function adds a repeat instance of the spike builder refereced by
        sb_id_val, which can be either an id or a name, at the specified start_time.

        Note that this doesnt validate whether this spike builder overlaps with
        another spike builder or not. That is done during preprocessing
        """
        if isinstance(sb_id_val, str):
            actual_id = self._spike_builders_name_map[sb_id_val]
        else:
            actual_id = int(sb_id_val)

        new_ri = (float(start_time), actual_id)
        if new_ri not in self._repeat_instances_set:
            self._repeat_instances_set = self._repeat_instances_set.union({new_ri})
        else:
            raise KeyError("The repeat instance {} is already contained in the spike builder".format(new_ri))

    @property_setter('repeat_instances')
    def pop_repeat_instance(self, start_time=None, sb_id_val=None):
        """
        This function removes a single spike generator specified by either the
        start_time and/or he sb_id_val (either id or name). If both are unspecified,
        a random repeat instance is popped
        """
        if start_time is None and sb_id_val is None:
            new_set = set(self._repeat_instances_set)
            popped_ri = new_set.pop
            self._repeat_instances_set = frozenset(new_set)
            return popped_ri
        else:
            repeat_instance_list = list(self._repeat_instances_set)

            if start_time is not None:
                repeat_instance_list = [x for x in repeat_instance_list if x[0] == float(start_time)]

            if sb_id_val is not None:
                if isinstance(sb_id_val, str):
                    actual_id = self._spike_builders_name_map[sb_id_val]
                else:
                    actual_id = int(sb_id_val)            
                repeat_instance_list = [x for x in repeat_instance_list if x[1] == actual_id]

            if repeat_instance_list:
                self._repeat_instances_set = self._repeat_instances_set.difference({repeat_instance_list[0]})
                return repeat_instance_list[0]
            else:
                raise KeyError("The required repeat instance was not found")

    @property_setter('repeat_instances')
    def clear_repeat_instances(self):
        self._repeat_instances_set = frozenset()

    LegacyPatternInfo = namedtuple('LegacyPatternInfo', ['spike_times',
                                                         'pattern_ind_t',
                                                         'pattern_seq',
                                                         'pattern_ind_p',
                                                         'pattern_start_p',
                                                         'pattern_length_p'])

    @requires_built
    def get_legacy_struct(self):
        """
        This returns the legacy tuple for use with christophs code. only in this case
        it is a named_tuple which has backward compatibility with the plain tuple
        """
        LegacyPatternInfo = RepeaterSpikeBuilder.LegacyPatternInfo
        spike_times = self.spike_time_array
        pattern_ind_t = -1*np.ones(self.steps_length)
        pattern_ind_p = np.zeros(len(self._repeat_instances_sorted))
        pattern_start_p = np.zeros(len(self._repeat_instances_sorted))
        pattern_length_p = np.zeros(len(self._repeat_instances_sorted))

        for i in range(len(_repeat_instances_sorted)):
            ri_entry = _repeat_instances_sorted[i]
            sb = self._spike_builders[ri_entry[1]]
            steps_per_ms = sb.steps_per_ms
            sb_start_time_step = int(ri_entry[0]*steps_per_ms + 0.5)
            sb_end_time_step = sb_start_time_step + sb.steps_length

            pattern_ind_t[sb_start_time_step:sb_end_time_step] = ri_entry[1]
            pattern_ind_p[i] = ri_entry[1]
            pattern_start_p[i] = (self._start_time_step + sb_start_time_step)/steps_per_ms
            pattern_length_p[i] = sb.time_length

        return LegacyPatternInfo(
            spike_times=spike_times,
            pattern_ind_t=pattern_ind_t,
            pattern_seq=None,
            pattern_ind_p=pattern_ind_p,
            pattern_start_p=pattern_start_p,
            pattern_length_p=pattern_length_p)