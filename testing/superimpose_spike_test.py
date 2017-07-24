__author__ = 'Arjun'

import ipdb

import numpy as np

with ipdb.launch_ipdb_on_exception():
    from spikebuilder.test_tools import analyse_exponential_distrib, getTotalIATVector
    from spikebuilder import CombinedSpikeBuilder
    from spikebuilder import RateBasedSpikeBuilder
    from ratebuilder import OURateBuilder


def main():
    # It is assumed that the following rate gen will almost never produce
    # negative values
    ou_rate_builder = OURateBuilder(mean=30, sigma=3, theta=0.5,
                                    steps_per_ms=1, time_length=80000, channels=[1, 2, 3])

    spike_pattern1 = RateBasedSpikeBuilder(rate_builder=ou_rate_builder.build_copy().set_frozen())
    spike_pattern2 = RateBasedSpikeBuilder(rate_builder=ou_rate_builder.build_copy().set_frozen())
    spike_pattern3 = RateBasedSpikeBuilder(rate_builder=ou_rate_builder.build_copy().set_frozen())
    spike_pattern4 = RateBasedSpikeBuilder(rate_builder=ou_rate_builder.build_copy().set_frozen(),)

    combined_spike_builder = CombinedSpikeBuilder(spike_builder_list=[spike_pattern1,
                                                                      spike_pattern2,
                                                                      spike_pattern3,
                                                                      spike_pattern4],
                                                  repeat_instances_list=[(0, 0),
                                                                         (80000, 1),
                                                                         (80000, 2),
                                                                         (160000, 3)])

    combined_spike_builder.build()

    assert np.all(combined_spike_builder.channels == spike_pattern1.channels)

    combined_rate_array = np.hstack((
        spike_pattern1.rate_builder.rate_array,
        spike_pattern2.rate_builder.rate_array + spike_pattern3.rate_builder.rate_array,
        spike_pattern4.rate_builder.rate_array))

    TotalIATVector = getTotalIATVector(combined_spike_builder, combined_rate_array)
    analyse_exponential_distrib(TotalIATVector)

    try:
        ou_rate_builder.steps_per_ms = 2
        spike_pattern3_half_step = spike_pattern3.copy()
        spike_pattern3_half_step.rate_builder = ou_rate_builder.build_copy().set_frozen()

        combined_spike_builder.spike_builders = [
            spike_pattern1,
            spike_pattern2,
            spike_pattern3_half_step,
            spike_pattern4
        ]
        combined_spike_builder.build()
    except AssertionError as E:
        E_msg = E.args[0]
        if 'consistent stepsize' in E_msg:
            print("Successfully caught inconsistent stepsize, generated following exception")
            print(E)
            combined_spike_builder.spike_builders = [spike_pattern1,
                                                     spike_pattern2,
                                                     spike_pattern3,
                                                     spike_pattern4]
            ou_rate_builder.steps_per_ms = 1
        else:
            raise
    except Exception as E:
        raise
    else:
        raise Exception("The combined spike builder processed an inconsistent stepsize")

    # Change pat3 to contain a new rate generator
    ou_rate_builder.channels = [4, 5, 6]
    spike_pattern3.rate_builder = ou_rate_builder.build_copy().set_frozen()
    curr_spike_builders = list(combined_spike_builder.spike_builders)
    curr_spike_builders[2] = spike_pattern3
    combined_spike_builder.spike_builders = curr_spike_builders

    combined_spike_builder.build()
    spike_builds = list(combined_spike_builder.spike_builders)

    # Calculate the new combined rate array for the new pattern superimposition
    combined_rate_array = np.hstack((
        np.vstack((spike_builds[0].rate_builder.rate_array, np.zeros((3, spike_builds[0].steps_length)))),
        np.vstack((spike_builds[1].rate_builder.rate_array, spike_builds[2].rate_builder.rate_array)),
        np.vstack((spike_builds[3].rate_builder.rate_array, np.zeros((3, spike_builds[3].steps_length))))
    ))

    # Analyse the spikes. If it works it means that comibining sifferent channel
    # and start time change works fine
    TotalIATVector = getTotalIATVector(combined_spike_builder, combined_rate_array)
    analyse_exponential_distrib(TotalIATVector)


if __name__ == '__main__':
    with ipdb.launch_ipdb_on_exception():
        main()
