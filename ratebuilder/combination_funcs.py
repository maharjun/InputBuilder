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

def combine_sigmoid(K, M):
    """
    This is a function that returns the appropriate sigmoid function to be used in
    CombinedRateBuilder. See code to find out what it does. it's simple enough
    """
    def combine_sigmoid_func(rate_array_list):
        assert len(rate_array_list) == 2, "sigmoidal combination is only valid for 2 rate arrays"
        A = rate_array_list[0]
        B = rate_array_list[1]
        f_max = max(np.amax(A), np.amax(B))
        x = A + B

        return f_max/(1+np.exp(-2*K*(x-1.0*M*f_max)/f_max))

    return combine_sigmoid_func