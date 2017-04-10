# combination_funcs.py
# 
#   Author: Arjun Rao
# 
# This file contains functions and metafunctions that perform the combination of two or more arrays
# to give a combined array. This is used to provide arguments to the CombinedRateBuilder.

import numpy as np

def combine_sum(rate_array_list):
    return_arr = 0
    for arr in rate_array_list:
        return_arr = return_arr + arr
    return return_arr

class combine_sigmoid:
    """
    This is a class that returns a callable object that represents the appropriate
    sigmoid function to be used in CombinedRateBuilder. See code to find out what
    it does. it's simple enough
    """

    def __init__(self, K, M):
        self.K = np.float(K)
        self.M = np.float(M)

    def __call__(self, rate_array_list):
        assert len(rate_array_list) == 2, "sigmoidal combination is only valid for 2 rate arrays"
        A = rate_array_list[0]
        B = rate_array_list[1]
        f_max = max(np.amax(A), np.amax(B))
        x = A + B

        K = self.K
        M = self.M

        return f_max/(1+np.exp(-2*K*(x-1.0*M*f_max)/f_max))