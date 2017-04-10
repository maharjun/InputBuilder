from . import BaseSpikeBuilder

from genericbuilder.propdecorators import requires_built

from numpy.random import mtrand
import numpy as np

mtgen = mtrand.binomial.__self__

class ConstRateSpikeBuilder(BaseSpikeBuilder):

    def __init__(self, rate,
                       channels=[], steps_per_ms=1, time_length=0,
                       rng=mtgen):

        super().__init__()

        self.rate = rate
        self.channels = channels
        self.steps_per_ms = steps_per_ms
        self.time_length = time_length
        self.rng = rng

    def _validate(self):
        pass

    def _preprocess(self):
        pass

    @property
    def time_length(self):
        """
        Returns the time length of the current spike pattern in ms

        :returns: an np.float64 value representing the time length of the spike pattern
        """
        return self._time_length

    @time_length.setter
    def time_length(self, time_length):
        assert time_length >= 0, "'time_length' must be a non-negative number"
        self._time_length = np.float64(time_length)

    @property
    def steps_per_ms(self):
        """
        Returns/sets the time resolution in terms of simulation steps per ms

        :returns: An np.uint32 that represents the
        """
        return self._steps_per_ms

    @steps_per_ms.setter
    def steps_per_ms(self, steps_per_ms):
        assert steps_per_ms >= 1, "'steps_per_ms' must be a positive integer"
        self._steps_per_ms = np.uint32(steps_per_ms)

    @property
    def channels(self):
        """
        Returns/sets the channels for which the spikes are generated.

        :returns: a numpyp 1-D uint32 array containing the indices of the channels in
            ascending order
        """
        return self._channels

    @channels.setter
    def channels(self, channels):
        channels = np.array(channels)
        channels = (channels + 0.5).astype(np.int32)  # 0.5 for rounding off
        channels_unique_sorted = np.lib.arraysetops.unique(channels)
        assert np.all(channels_unique_sorted >= 0), "Channel indices should be non-negative integers"
        channels_unique_sorted = channels_unique_sorted.astype(np.uint32)
        channels_unique_sorted.setflags(write=False)
        self._channels = channels_unique_sorted

    @property
    @requires_built
    def spike_rel_step_array(self):
        """
        Property that returns the relative spike step array.

        :returns: an array of arrays A such that::
            
              A[i][j] = TIME STEP of the jth spike of the ith neuron relative to the
                        beginning of the spike pattern
        """
        return self._spike_rel_step_array

    @property
    @requires_built
    def spike_weight_array(self):
        """
        Property that returns the spike weight array.

        :returns: an array of arrays A such that::
            
              A[i][j] = WEIGHT of the jth spike of the ith neuron
        """
        return self._spike_weight_array

    @property
    def rate(self):
        return self._rate

    @rate.setter
    def rate(self, rate_val):
        if rate_val >= 0:
            self._rate = float(rate_val)
        else:
            raise ValueError("'rate' must be a non-negative float")

    @property
    def rng(self):
        return self._rng

    @rng.setter
    def rng(self, rng_):
        self._rng = rng_

    def _build(self):
        super()._build()
        nchannels = self.channels.size
        spike_weight_array = self._rng.poisson(lam=self.rate/(1000*self.steps_per_ms),
                                        size=(nchannels, self.steps_length)).astype(np.uint32)
        self._spike_rel_step_array = np.ndarray(nchannels, dtype=object)
        self._spike_weight_array = np.ndarray(nchannels, dtype=object)

        for i in range(nchannels):
            self._spike_rel_step_array[i] = np.argwhere(spike_weight_array[i,:])[:,0].astype(np.uint32)
            self._spike_weight_array[i] = spike_weight_array[i, self._spike_rel_step_array[i]]
            self._spike_rel_step_array[i].setflags(write=False)
            self._spike_weight_array[i].setflags(write=False)
        self._spike_rel_step_array.setflags(write=False)
        self._spike_weight_array.setflags(write=False)