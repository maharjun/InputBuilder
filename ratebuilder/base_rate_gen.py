__author__ = 'Arjun'

from genericbuilder.baseclass_simple import BaseGenericBuilder
from genericbuilder.propdecorators import requires_built

from abc import abstractmethod

class BaseRateBuilder(BaseGenericBuilder):

    builder_type = 'rate'

    def __init__(self):
        """Constructor for BaseRateBuilder

        The constructor takes one argument either named copy_object or conf_dict.

        If it is copy_object, the object must be an instance of BaseRateBuilder. In
        this case, init acts like a copy constructor.

        If it is config_dict, the argument must be dictionary subscriptable. The
        relevant keys are

        1.  steps_per_ms
        2.  time_length
        3.  channels

        When the above keys are not specified, they are initialized to their default
        values
        """
        super().__init__()  # initializes the core builder flags

        # Default Initialization of data members with defaults

        # Assignment of properties from the parameters
        # self.steps_per_ms = conf_dict.get('steps_per_ms')
        # self.time_length = conf_dict.get('time_length')
        # self.channels = conf_dict.get('channels')

    # ------------------------------------------------------------------------------------ #
    # CORE INTERFACE
    # ------------------------------------------------------------------------------------ #
    #
    # The core interface methods and properties must be defined in the subclass rate generator

    # These 3 functions are abstract functions that need to be implemented. They
    # are defined in BaseGenericBuilder.

    # def _build(self):
    # def _preprocess(self):
    # def _validate(self):

    @property
    @abstractmethod
    def steps_per_ms(self):
        """
        Get or Set the time resolution of the rate pattern by specifying an integer
        representing the number of time steps per ms

        :return: np.uint32 giving the number of simulation steps per ms
        """
        pass

    @property
    @abstractmethod
    def time_length(self):
        """
        Get or set the time length of the rate pattern in ms.

        :GET: This function will return the time length of the built rate pattern i.e.
            the unrounded time length. not `self._steps_length/self.steps_per_ms`

        :SET: This function will round the time to the nearest time step and use that
            as the actual time length.

        :return: An np.float64 scalar
        """
        pass

    @property
    @abstractmethod
    def channels(self):
        """
        Returns read-only view of channels property. In order to make it writable, copy
        it via X.channels.copy() or np.array(X.channels)

        :returns: a 1-D unwritable numpy array with channel indices sorted in ascending
            order
        """
        channels_view = self._channels[:]
        channels_view.setflags(write=False)
        return channels_view

    @property
    @requires_built
    @abstractmethod
    def rate_array(self):
        """
        Returns read-only view of channels property. In order to make it writable, copy
        it via X.rate_array.copy() or np.array(X.rate_array)
        """
        pass