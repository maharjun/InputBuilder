__author__ = 'Arjun'

import numpy as np
from numba import jit
from scipy.stats import kstest


@jit("double[:](double[:], uint32, uint32[:])", cache=True, nopython=True)
def convert_poisson_seq_to_IAT(rate_seq, steps_per_ms, poisson_sampling_seq):
    """
    This function does the following:

    If the poisson sampling sequence is accuraely sampled from the given rate
    sequence (Hz), this function should return a sequence of IAS that are
    distributed as exp(2). Why 2 because 1^2 = 1 (var = mean^2) and that would
    cover up some errors I guess
    """
    base_rate = 1/2
    time_stretch_factor = rate_seq/base_rate

    spike_IAT_seq = []
    prev_spike_interval = 0
    for i in range(len(time_stretch_factor)):
        Nspikes = poisson_sampling_seq[i]
        current_interval = time_stretch_factor[i]/(1000*steps_per_ms)
        if Nspikes == 0:
            prev_spike_interval += current_interval
        else:
            random_spiketimes = np.random.uniform(0.0, current_interval, Nspikes)
            random_spiketimes = np.sort(random_spiketimes)
            spike_IAT_seq.append(random_spiketimes[0] + prev_spike_interval)
            spike_IAT_seq += list(np.diff(random_spiketimes))
            prev_spike_interval = current_interval - random_spiketimes[-1]

    return np.array(spike_IAT_seq)


def analyse_exponential_distrib(data, expected_mean=2, error_bound=0.05):
    """
    Checks if he data is exp distributed with mean expected_mean.
    """

    print("Statistics for estimated dstribution")

    mean_est = np.mean(data)
    var_est = np.mean((data - expected_mean)**2)
    print("Mean    : {:<10.5f} Expected: {:<10.5f}".format(mean_est, expected_mean))
    print("Variance: {:<10.5f} Expected: {:<10.5f}".format(var_est, expected_mean**2))

    D, p_value = kstest(data, 'expon', (0, expected_mean))
    assert D < error_bound, \
        "The Mean And Variance are not within the {:.0f}% bounds".format(error_bound*100)
    return (mean_est, var_est)


def getTotalIATVector(spike_builder, rate_array):

    return getTotalIATVectorFrom(spike_builder.spike_rel_step_array,
                                 spike_builder.steps_per_ms,
                                 rate_array,
                                 spike_builder.spike_weight_array)


def getTotalIATVectorFrom(spike_rel_step_array,
                          steps_per_ms,
                          rate_array,
                          spike_weight_array=None):

    steps_length = rate_array.shape[1]

    if spike_weight_array is None:
        spike_weight_array = [np.ones_like(x, dtype=np.uint32) for x in spike_rel_step_array]
        spike_weight_array = np.array(spike_weight_array, dtype=object)

    assert rate_array.shape[0] == spike_rel_step_array.shape[0]
    assert all(x.size == y.size for x, y in zip(spike_rel_step_array, spike_weight_array))
    assert all(np.all(x < steps_length) for x in spike_rel_step_array)

    IATVectors = []
    for i, non_zero_spike_inds in enumerate(spike_rel_step_array):
        poisson_sampling_sequence = np.zeros(steps_length, dtype=np.uint32)
        poisson_sampling_sequence[non_zero_spike_inds] = spike_weight_array[i]
        IATVectors.append(convert_poisson_seq_to_IAT(rate_array[i, :],
                                                     steps_per_ms,
                                                     poisson_sampling_sequence))
        TotalIATVector = np.concatenate(IATVectors)
        # print("Finished Channel Index ", i)
    return TotalIATVector
