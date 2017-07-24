import numpy as np
import ipdb

with ipdb.launch_ipdb_on_exception():
    from ratebuilder import CombinedRateBuilder
    from ratebuilder import OURateBuilder
    from ratebuilder import LegacyRateBuilder
    from ratebuilder.comb_rate_gen import combine_sigmoid

    import matplotlib
    matplotlib.use('Agg')

    import matplotlib.pyplot as plt


def test1():
    """
    ASSUMPTION:
    OURateBuilder is tested and working

    TEST:
    Here we simply take additive combination of 2 OURateBuilders with the same theta
    and measure the statistics.
    """
    steps_per_ms = 2
    channels = range(0, 100)
    time_length = 5000

    sim_params = {
        'steps_per_ms': steps_per_ms,
        'channels': channels,
        'time_length': time_length
    }

    ou_rate_bldr1 = OURateBuilder(**dict(sim_params, mean=20, sigma=2, theta=1))
    ou_rate_bldr2 = OURateBuilder(**dict(sim_params, mean=30, sigma=3, theta=1))

    comb_rate_bldr = CombinedRateBuilder(rate_builders=[ou_rate_bldr1, ou_rate_bldr2])
    comb_rate_bldr.time_length = 10000

    comb_rate_bldr.build()
    # Check if combination successful

    assert np.all(comb_rate_bldr.rate_array ==
                  comb_rate_bldr.rate_builders[0].rate_array + comb_rate_bldr.rate_builders[1].rate_array), \
        "The combination was unsuccessful"
    print("The combination was successful")


def test2():
    """
    ASSUMPTION
    The LegacyRateBuilder is tested and working correctly

    TESTS:
    Tests that the sigmoidal combination here gives the same output as in christophs code. A
    sample output is included in the directory
    """

    steps_per_ms = 1
    channels = range(0, 10)
    time_length = 200000

    sim_params = {
        'steps_per_ms': steps_per_ms,
        'channels': channels,
        'time_length': time_length
    }

    leg_builder1 = LegacyRateBuilder(**dict(sim_params,
                                            mean=np.log(3), sigma=0.5, theta=0.005,
                                            delay=50, max_rate=50))
    leg_builder2 = leg_builder1.copy()

    comb_rate_bldr = CombinedRateBuilder(
        rate_builders=[leg_builder1, leg_builder2],
        transform=combine_sigmoid(K=5.0, M=0.5))

    comb_rate_bldr.build()
    comb_rate_hist_fig = plt.figure()

    ax = plt.gca()
    ax.hist(comb_rate_bldr.rate_array.ravel(), 200, color='green', edgecolor='green', log=True)
    ax.set_xlabel('Rate (Hz)')
    ax.set_ylabel('Count')
    ax.set_title('Mean=%.2fHz, SD=%.2fHz' % (np.mean(comb_rate_bldr.rate_array),
                                             np.std(comb_rate_bldr.rate_array, ddof=1)))
    comb_rate_hist_fig.savefig('combined_rate_hist.png', format='png', dpi=300)


if __name__ == '__main__':
    with ipdb.launch_ipdb_on_exception():
        print("Starting test 1")
        test1()
        print("Completed Test 1")
        print("")

        print("Starting Test 2")
        print("For this test, compare the generated histogram and statistics in combined_rate_hist.png\n"
              "to those in the provided legacy_hist.png")
        test2()
        print("Completed Test 2")
