import numpy as np

from genericbuilder.propdecorators import *

from . import SuperimposeSpikeBuilder

class DisjointSpikeBuilder(SuperimposeSpikeBuilder):
    """
    Disjoint Spike builder is a sub-class of SuperimposeSpikeBuilder. The differences
    are that the validation imposes the additional condition that the constituent
    patterns are disjoint. It also contains some additional gettable properties such
    as legacy struct to keepp compatibility with old pattern generators etc.

    Initialization and everything else is identical to SuperimposeSpikeBuilder
    """

    def __init__(self, conf_dict):
        super().__init__(conf_dict)

    def _validate(self):
        super()._validate()
        start_end_time_list = [(sb.start_time, sb.start_time + sb.time_length)
                               for sb in self._spike_builders.values()]
        start_end_time_list = sorted(start_end_time_list)
        is_disjoint_list = [start_end_time_list[i+1][0] >= start_end_time_list[i][1]
                            for i in range(len(start_end_time_list) - 1)]
        is_disjoint = all(is_disjoint_list)
        assert is_disjoint, "All the constituent spike builders must correspond to disjoint intervals"

    def _build(self):
        super()._build()

    @property
    @requires_preprocessed
    def sorted_patterns(self):
        return sorted(self._spike_builders_list, key=lambda x: x.start_time)
