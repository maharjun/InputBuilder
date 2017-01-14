__author__ = 'Arjun'

import ipdb

with ipdb.launch_ipdb_on_exception():
    from spikebuilder.test_tools import analyse_exponential_distrib, getTotalIATVector
    from spikebuilder import RepeaterSpikeBuilder
    from spikebuilder import RateBasedSpikeBuilder
    from spikebuilder import ConstRateSpikeBuilder
    from ratebuilder import OURateBuilder

import numpy as np

def main():
    # It is assumed that the following rate gen will almost never produce
    # negative values
    steps_per_ms = 1
    ou_rate_builder = OURateBuilder({
        'steps_per_ms': steps_per_ms,
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
    
    combined_spike_builder = RepeaterSpikeBuilder({
        'spike_builders': [('pat1', spike_pattern1),
                           ('pat2', spike_pattern2),]})

    combined_spike_builder.add_repeat_instance(0, 0)
    combined_spike_builder.add_repeat_instance(80000, 0)
    combined_spike_builder.add_repeat_instance(30000, 0)
    
    try:
        combined_spike_builder.build()
    except AssertionError as E:
        if E.args[0] == "All the constituent spike builders must correspond to disjoint intervals":
            print("SUCCESSFULLY caught the following exception:")
            print(E.args[0])
        else:
            raise E
    else:
        raise Exception("Unsuccessfully processed overlapping inputs")

    combined_spike_builder.pop_repeat_instance(30000)

    combined_spike_builder.add_repeat_instance(160000, 0)

    combined_spike_builder.build()
    assert np.all(combined_spike_builder.channels == spike_pattern1.channels)

    combined_rate_array = np.hstack((
        spike_pattern1.rate_builder.rate_array,
        spike_pattern2.rate_builder.rate_array,
        spike_pattern1.rate_builder.rate_array))

    base_rate = combined_spike_builder.spike_builders['pat1'].rate_builder.get_properties()['mean']
    TotalIATVector = getTotalIATVector(combined_spike_builder, combined_rate_array, base_rate)
    analyse_exponential_distrib(TotalIATVector, expected_mean=1/base_rate)

    # Now adding fillers.
    combined_spike_builder.clear_repeat_instances()
    combined_spike_builder.add_repeat_instance(0, 0)
    combined_spike_builder.add_repeat_instance(100000, 1)
    combined_spike_builder.add_repeat_instance(200000, 0)
    
    prior_props = ou_rate_builder.get_properties()
    prior_props.update({'rate': 5})
    filler_spike_builder = ConstRateSpikeBuilder(prior_props)
    combined_spike_builder.add_spike_builder(filler_spike_builder, 'filler')

    combined_spike_builder.build()
    combined_rate_array = np.hstack((
        spike_pattern1.rate_builder.rate_array,
        filler_spike_builder.rate*np.ones((spike_pattern1.channels.size, 20000*steps_per_ms)),
        spike_pattern2.rate_builder.rate_array,
        filler_spike_builder.rate*np.ones((spike_pattern1.channels.size, 20000*steps_per_ms)),
        spike_pattern1.rate_builder.rate_array))

    TotalIATVector = getTotalIATVector(combined_spike_builder, combined_rate_array, base_rate)
    analyse_exponential_distrib(TotalIATVector, expected_mean=1/base_rate)



if __name__ == '__main__':
    with ipdb.launch_ipdb_on_exception():
        main()