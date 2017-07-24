from . import BaseRateBuilder
from genericbuilder.propdecorators import requires_built
import numpy as np


class ConstRateBuilder(BaseRateBuilder):

    def __init__(self, rate,
                 channels=[], steps_per_ms=1, time_length=0):
        super().__init__()  # only purpose is to run BaseGenericBuilder init

        self.rate = rate
        self.channels = channels
        self.steps_per_ms = steps_per_ms
        self.time_length = time_length

    def _validate(self):
        pass

    def _preprocess(self):
        self._steps_length = np.uint32(self._time_length * self._steps_per_ms + 0.5)

    @property
    def rate(self):
        return self._rate

    @rate.setter
    def rate(self, rate):
        assert rate >= 0, "'rate' must be a non-negative number"
        self._rate = rate

    @property
    def steps_per_ms(self):
        """
        Get or Set the time resolution of the rate pattern by specifying an integer representing
        the number of time steps per ms

        :return:
        """
        return self._steps_per_ms

    @steps_per_ms.setter
    def steps_per_ms(self, steps_per_ms_):
        if steps_per_ms_ is None:
            self._init_attr('_steps_per_ms', np.uint32(1))
        else:
            if steps_per_ms_ >= 1:
                self._steps_per_ms = np.uint32(steps_per_ms_)
            else:
                raise ValueError("'steps_per_ms' must be non-zero positive integer")

    @property
    def time_length(self):
        """
        Get or set the time length of the rate pattern in ms.

        :GET: This function will return the time length of the built rate pattern i.e. the
            rounded time length. i.e. `self._steps_length/self.steps_per_ms`

        :SET: This function will round the time to the nearest time step and use that as the actual
            time length.

        :return: An np.float64 scalar
        """
        return self._steps_length / self._steps_per_ms

    @time_length.setter
    def time_length(self, time_length_):
        if time_length_ is None:
            self._init_attr('_time_length', np.float64(0))
        else:
            if time_length_ >= 0:
                self._time_length = np.float64(time_length_)
            else:
                raise ValueError("property 'time_length' must be a non-negative numeric value")

    @property
    def channels(self):
        """
        Returns channels property. In order to make it writable, copy it via X.channels.copy() or
        np.array(X.channels)
        """
        return self._channels

    @channels.setter
    def channels(self, channels_):
        # Assuming 1D iterable
        if channels_ is None:
            self._init_attr('_channels', np.ndarray(0, dtype=np.uint32))
        else:
            channel_unique_array = np.array(list(set(channels_)), dtype=np.int32)
            if np.all(channel_unique_array >= 0):
                self._channels = np.array(channel_unique_array, dtype=np.uint32)
                self._channels.setflags(write=False)
            else:
                raise ValueError("'channels' must be a vector of non-negative integers")

    @property
    @requires_built
    def rate_array(self):
        return self._rate_array

    def _build(self):
        self._rate_array = self._rate
