from . import CombinedSpikeBuilder
from genericbuilder.propdecorators import *

class RepeaterSpikeBuilder(CombinedSpikeBuilder):
    """
    NEED TO WRITE DOCUMENTATION HERE
    """
    def __init__(self, conf_dict):
        self._repeat_instances_set = set()
        self._repeat_instances_sorted = []  # derived from 
        self._is_time_length_assigned = False

        conf_dict_base = {key:conf_dict.get(key) for key in ['spike_builders', 'start_time']}

        super().__init__(conf_dict_base)
        self.time_length = conf_dict.get('time_length') or None


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
        actual_spike_builders_list = [self._spike_builders[sb_uid].with_start_time(sb_st).copy()
                                      for sb_st, sb_uid in self._repeat_instances_sorted]
        
        # Adding Filler Spike Builders if present
        filler_spike_builders_list = []
        if 'filler' in self._spike_builders_name_map:
            filler_uid = self._spike_builders_name_map['filler']
            all_start_end_times = [(sb_st, self._spike_builders[sb_uid].time_length + sb_st)
                                   for sb_st, sb_uid in self._repeat_instances_sorted]
            filler_start_end_times = [(all_start_end_times[i][1], all_start_end_times[i+1][0])
                                      for i in range (len(self._repeat_instances_sorted)-1)]

            filler_spike_builder_copy = self._spike_builders[filler_uid].copy()
            filler_spike_builders_list = [
                filler_spike_builder_copy.with_start_time(fill_beg)
                                         .with_time_length(fill_end - fill_beg)
                                         .copy()
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
    @requires_rebuild
    def time_length(self, time_length_):
        if time_length_ is not None:
            super().with_time_length(time_length_)
            self._is_time_length_assigned = True
        else:
            self._is_time_length_assigned = False

    @property_setter('repeat_instances')
    @requires_rebuild
    def add_repeat_instance(self, start_time, sb_id_val):
        if isinstance(sb_id_val, str):
            actual_id = self._spike_builders_name_map[sb_id_val]
        else:
            actual_id = int(sb_id_val)

        new_ri = (float(start_time), actual_id)
        if new_ri not in self._repeat_instances_set:
            self._repeat_instances_set.add(new_ri)
        else:
            raise KeyError("The repeat instance {} is already contained in the spike builder".format(new_ri))

    @property_setter('repeat_instances')
    @requires_rebuild
    def pop_repeat_instance(self, start_time=None, sb_id_val=None):
        
        if start_time is None and sb_id_val is None:
            return self._repeat_instances_set.pop
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
                self._repeat_instances_set.remove(repeat_instance_list[0])
                return repeat_instance_list[0]
            else:
                raise KeyError("The required repeat instance was not found")

    @property_setter('repeat_instances')
    @requires_rebuild
    def clear_repeat_instances(self):
        self._repeat_instances_set.clear()
