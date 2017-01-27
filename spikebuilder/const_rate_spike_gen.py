from . import BaseSpikeBuilder

from numpy.random import mtrand as mt
import numpy as np

class ConstRateSpikeBuilder(BaseSpikeBuilder):


    def __init__(self, config_dict):

        super().__init__(config_dict)
        self.rate = None
        self.rng = None

        self.rate = config_dict.get('rate')
        self.rng = config_dict.get('rng')

    @property
    def rate(self):
        return self._rate

    @rate.setter
    def rate(self, rate_val):
        if rate_val is None:
            self._init_attr('_rate', 0.0)
        else:
            if rate_val >= 0:
                self._rate = float(rate_val)
            else:
                raise ValueError("'rate' must be a non-negative float")

    @property
    def rng(self):
        return self._rng

    @rng.setter
    def rng(self, rng_):
        if rng_ is None:
            self._init_attr('_rng', mt)
        else:
            self._rng = rng_

    def _build(self):
        super()._build()
        nchannels = self._channels.size
        spike_weight_array = mt.poisson(lam=self._rate/(1000*self._steps_per_ms),
                                        size=(nchannels, self._steps_length)).astype(np.uint32)
        self._spike_rel_step_array = np.array(
            [np.argwhere(spike_weight_array[i,:])[:,0].astype(np.uint32) for i in range(nchannels)],
            dtype=object)
        self._spike_weight_array = np.array(
            [spike_weight_array[i, self._spike_rel_step_array[i]] for i in range(nchannels)],
            dtype=object)