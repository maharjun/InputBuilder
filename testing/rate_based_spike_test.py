__author__ = 'Arjun'

import matplotlib.pyplot as plt
import time

from ratebuilder import OURateBuilder
from spikebuilder import RateBasedSpikeBuilder

import numpy as np

from spikebuilder.test_tools import convert_poisson_seq_to_IAT, analyse_exponential_distrib

import ipdb

def main():

    ou_gen = OURateBuilder(
        steps_per_ms=2,
        time_length=10000000,
        channels=[1],
        mean=2,
        sigma=1,
        theta=1)
    
    spike_gen = RateBasedSpikeBuilder(
        rate_builder=ou_gen,
        transform=np.abs
        )

    before = time.clock()
    spike_gen.build()
    after = time.clock()
    print("Completed Spike Generation")
    print("Time taken = {:<10.5f} seconds".format(after-before))

    rate_seq = spike_gen._transform(spike_gen.rate_builder.rate_array[0, :])
    nspike_seq = np.zeros(spike_gen.steps_length, dtype=np.uint32)
    nonzero_spike_inds = spike_gen.spike_rel_step_array[0]
    nspike_seq[nonzero_spike_inds] = spike_gen.spike_weight_array[0]

    IAT_seq = convert_poisson_seq_to_IAT(rate_seq,
                                         spike_gen.rate_builder.mean,
                                         nspike_seq)

    analyse_exponential_distrib(IAT_seq, 2)
    plt.hist(IAT_seq, 100)
    plt.show()

if __name__ == '__main__':
    with ipdb.launch_ipdb_on_exception():
        main()