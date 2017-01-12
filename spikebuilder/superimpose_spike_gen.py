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
          as calculated by CombinedSpikeBuilder

        steps_per_ms:
          as calculated by CombinedSpikeBuilder
        
        channels: 
          as calculated by CombinedSpikeBuilder

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
        
        super().__init__(conf_dict)

        self.start_time = conf_dict.get('start_time')


    def _preprocess(self):
        """
        Calculate time_length, start_time, from the spike_builders.
        These functions mimic the property setting functions of the 
        """
        self._final_spike_builders_list = list(self._spike_builders.values())
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