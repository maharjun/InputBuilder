__author__ = 'Arjun'

import ipdb

with ipdb.launch_ipdb_on_exception():
    from spikebuilder.test_tools import analyse_exponential_distrib, getTotalIATVector
    from spikebuilder import SuperimposeSpikeBuilder
    from spikebuilder import RateBasedSpikeBuilder
    from ratebuilder import OURateBuilder

import numpy as np

def main():
    # It is assumed that the following rate gen will almost never produce
    # negative values
    ou_rate_builder = OURateBuilder({
        'steps_per_ms': 1,
        'time_length' : 80000,
        'channels'    : [1, 2, 3],
        'mean': 30,
        'sigma': 3,
        'theta': 0.5
        })

    spike_pattern1 = RateBasedSpikeBuilder({
        'rate_builder': ou_rate_builder.build().copy_frozen(),
        'start_time': 0})
    spike_pattern2 = RateBasedSpikeBuilder({
        'rate_builder': ou_rate_builder.build().copy_frozen(),
        'start_time': 80000})
    spike_pattern3 = RateBasedSpikeBuilder({
        'rate_builder': ou_rate_builder.build().copy_frozen(),
        'start_time': 80000})
    spike_pattern4 = RateBasedSpikeBuilder({
        'rate_builder': ou_rate_builder.build().copy_frozen(),
        'start_time': 160000})

    combined_spike_builder = SuperimposeSpikeBuilder({
        'spike_builders': [('pat1', spike_pattern1),
                           ('pat2', spike_pattern2),
                           ('pat3', spike_pattern3),
                           ('pat4', spike_pattern4)]})

    combined_spike_builder.build()

    assert np.all(combined_spike_builder.channels == spike_pattern1.channels)

    combined_rate_array = np.hstack((
        spike_pattern1.rate_builder.rate_array,
        spike_pattern2.rate_builder.rate_array + spike_pattern3.rate_builder.rate_array,
        spike_pattern4.rate_builder.rate_array))

    base_rate = combined_spike_builder.spike_builders['pat1'].rate_builder.get_properties()['mean']
    TotalIATVector = getTotalIATVector(combined_spike_builder, combined_rate_array, base_rate)
    analyse_exponential_distrib(TotalIATVector, expected_mean=1/base_rate)

    # Try to change the channels of spike_pattern3. catch error and confirm
    try:
        combined_spike_builder.modify_spike_builder('pat3', {'channels': [4, 5, 6]})
    except ValueError as E:
        if E.args[0] == "Frozen rate builder cannot modify 'channels'":
            print("Successfully generated following exception")
            print(E)
        else:
            raise E
    else:
        raise Exception("The combined spike builder changed a frozen generator")

    try:
        combined_spike_builder.modify_spike_builder('pat3', 
            {'rate_builder': ou_rate_builder.with_steps_per_ms(2).build().copy_frozen()})
        combined_spike_builder.build()
    except AssertionError as E:
        print("Successfully caught inconsistent stepsize, generated following exception")
        print(E)
        combined_spike_builder.modify_spike_builder('pat3', spike_pattern3)
        ou_rate_builder.steps_per_ms = 1
    else:
        raise Exception("The combined spike builder processed an inconsistent stepsize")

    # Change pat3 to contain a new rate generator
    combined_spike_builder.modify_spike_builder('pat3', {
        "rate_builder": ou_rate_builder.with_channels([4, 5, 6]).build().copy_frozen()})
    combined_spike_builder.start_time = 30000
    combined_spike_builder.build()
    spike_builds = combined_spike_builder.spike_builders

    # Calculate the new combined rate array for the new pattern superimposition
    combined_rate_array = np.hstack((
        np.vstack((spike_builds['pat1'].rate_builder.rate_array, np.zeros((3, spike_builds['pat1'].steps_length)))),
        np.vstack((spike_builds['pat2'].rate_builder.rate_array, spike_builds['pat3'].rate_builder.rate_array)),
        np.vstack((spike_builds['pat4'].rate_builder.rate_array, np.zeros((3, spike_builds['pat4'].steps_length))))
        ))

    # Analyse the spikes. If it works it means that comibining sifferent channel
    # and start time change works fine
    TotalIATVector = getTotalIATVector(combined_spike_builder, combined_rate_array, base_rate)
    analyse_exponential_distrib(TotalIATVector, expected_mean=1/base_rate)

    # Confirm Start time change by checking the start time change in each of the contained generators
    new_start_times = set(int(gen.start_time+0.5) for gen in combined_spike_builder.spike_builders.values())
    assert new_start_times == {0, 80000, 160000}, "The Start Times have been unsuccesfully changed"
    print("The Start Times have been SUCCESSfully NOT changed")


if __name__ == '__main__':
    with ipdb.launch_ipdb_on_exception():
        main()