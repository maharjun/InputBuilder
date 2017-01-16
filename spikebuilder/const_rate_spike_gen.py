from . import BaseSpikeBuilder
from genericbuilder.propdecorators import requires_rebuild

from numpy.random import mtrand as mt
import numpy as np

class ConstRateSpikeBuilder(BaseSpikeBuilder):

    _shallow_copied_vars = {'_rng'}

    def __init__(self, config_dict):
        self._rate = 0

        super().__init__(config_dict)
        self.rate = config_dict.get('rate') or 0
        self.rng = config_dict.get('rng') or mt

    @property
    def rate(self):
        return self._rate

    @rate.setter
    @requires_rebuild
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
        nchannels = self._channels.size
        spike_weight_array = mt.poisson(lam=self._rate/(1000*self._steps_per_ms),
                                        size=(nchannels, self._steps_length)).astype(np.uint32)
        self._spike_rel_step_array = np.array(
            [np.argwhere(spike_weight_array[i,:])[:,0].astype(np.uint32) for i in range(nchannels)],
            dtype=object)
        self._spike_weight_array = np.array(
            [spike_weight_array[i, self._spike_rel_step_array[i]] for i in range(nchannels)],
            dtype=object)