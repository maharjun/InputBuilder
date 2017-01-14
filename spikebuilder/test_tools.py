__author__ = 'Arjun'

import numpy as np

def convert_poisson_seq_to_IAT(rate_seq, steps_per_ms, base_rate, poisson_sampling_seq):
    """
    This function does the following:

    base rate = 1 Hz
    """

    time_stretch_factor = rate_seq/base_rate

    spike_IAT_seq = []
    prev_spike_interval = 0
    for i in range(len(time_stretch_factor)):
        Nspikes = poisson_sampling_seq[i]
        current_interval = time_stretch_factor[i]/(1000*steps_per_ms)
        if Nspikes == 0:
            prev_spike_interval += current_interval
        else:
            random_spiketimes = np.random.uniform(low=0.0, high=current_interval, size=Nspikes)
            random_spiketimes = np.sort(random_spiketimes)
            spike_IAT_seq.append(random_spiketimes[0] + prev_spike_interval)
            spike_IAT_seq += list(np.diff(random_spiketimes))
            prev_spike_interval = current_interval - random_spiketimes[-1]

    return np.array(spike_IAT_seq)


def analyse_exponential_distrib(data, expected_mean):

    print("Statistics for estimated dstribution")
    print("Mean    : {:<10.5f} Expected: {:<10.5f}".format(np.mean(data), expected_mean))
    print("Variance: {:<10.5f} Expected: {:<10.5f}".format(
        np.mean((data - expected_mean)**2),
        expected_mean**2))


def getTotalIATVector(spike_builder, rate_array, base_rate):
    assert rate_array.shape[1] == spike_builder.steps_length

    IATVectors = []
    for i in range(len(spike_builder.channels)):
        poisson_sampling_sequence = np.zeros((spike_builder.steps_length,), dtype=np.uint32)
        non_zero_spike_inds = spike_builder.spike_rel_step_array[i]
        poisson_sampling_sequence[non_zero_spike_inds] = spike_builder.spike_weight_array[i]
        IATVectors.append(convert_poisson_seq_to_IAT(
                              rate_array[i,:],
                              spike_builder.steps_per_ms,
                              base_rate,
                              poisson_sampling_sequence))
        TotalIATVector = np.concatenate(IATVectors)
        print("Finished Channel {}".format(spike_builder.channels[i]))
    return TotalIATVector
