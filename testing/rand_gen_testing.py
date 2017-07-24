from ratebuilder import OURateBuilder
from spikebuilder import RateBasedSpikeBuilder

from numpy import random
import numpy as np
import copy
import ipdb


def main():
    global_rng = random.RandomState(30)

    ou_gen = OURateBuilder(mean=30, sigma=2, theta=1,
                           steps_per_ms=1, time_length=30000, channels=[1, 2, 3],
                           rng=global_rng)

    spike_gen = RateBasedSpikeBuilder(rate_builder=ou_gen, rng=global_rng)

    assert ou_gen.rng is spike_gen.rate_builder.rng is spike_gen.rng, "The Random generators are not being shared"

    print("Random generators being successfully shared")

    ou_gen.build()
    spike_gen.build()

    ou_gen_temp_result = ou_gen.rate_array.copy()
    spike_gen_temp_result = copy.deepcopy(spike_gen.spike_rel_step_array)

    global_rng.seed(30)

    ou_gen.build()
    spike_gen.build()

    ou_gen_temp_result1 = ou_gen.rate_array.copy()
    spike_gen_temp_result1 = copy.deepcopy(spike_gen.spike_rel_step_array)

    assert np.all(ou_gen_temp_result == ou_gen_temp_result1), "The results are inconsistent"
    assert all(np.all(spike_gen_temp_result[i] == spike_gen_temp_result1[i])
               for i in range(len(spike_gen.channels))), "The results are inconsistent"
    print("The results are consistent")


if __name__ == '__main__':
    with ipdb.launch_ipdb_on_exception():
        main()
