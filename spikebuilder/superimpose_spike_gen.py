__author__ = 'Arjun'

from . import CombinedSpikeBuilder
from genericbuilder.propdecorators import *
from genericbuilder.tools import *

class SuperimposeSpikeBuilder(CombinedSpikeBuilder):
    """
    This spike builder superimposes multiple spike trains on each other to create
    a resultant spike train. This inherits from CombinedSpikeBuilder. 
    
    This class may be inherited from in order to create a builder that performs some
    sort of superimposed spike building. (Note: Superimposed spike building means
    building  by linearly combining the output of different spike builders). The
    spike builders that are actually used for the building can be derived in some
    manner from the spike builders in self._spike_builders and stored as a list in
    self._final_spike_builders_list. In SuperimposeSpikeBuilder, we simply make

      self._final_spike_builders_list = list(self._spike_builders.values())
    
    2.  The common properties are calculated from the input spike trains as follows

        time_length
          dependent variable (UNSETTABLE). Calculated as:

          ``max(sb.start_time + sb.time_length)``  where sb is contained spike builder

        steps_per_ms:
          as calculated by CombinedSpikeBuilder
        
        channels: 
          as calculated by CombinedSpikeBuilder


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
        conf_dict = {key:conf_dict.get(key) for key in ['spike_builders', 'start_time']}
        super().__init__(conf_dict)

    def _preprocess(self):
        """
        Calculate time_length, start_time, from the spike_builders.
        These functions mimic the property setting functions of the 
        """
        self._final_spike_builders_list = list(self._spike_builders.values())
        if self._final_spike_builders_list:
            super().with_time_length(max(sb.start_time + sb.time_length for sb in self._final_spike_builders_list))
        super()._preprocess()

    # Making time_length unsettable
    time_length = CombinedSpikeBuilder.time_length.setter(None)

    def _build(self):
        super()._build()