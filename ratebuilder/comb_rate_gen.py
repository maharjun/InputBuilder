from . import BaseRateBuilder
from genericbuilder.propdecorators import *
from genericbuilder.tools import get_builder_type, get_unsettable

import numpy as np

class CombinedRateBuilder(BaseRateBuilder):

    def __init__(self, conf_dict):
        # Initialization done directly as no None initializable property / function
        # corresponding to _rate_builders
        self._rate_builders = []
        super().__init__({})

        relevant_keys = ['rate_builders', 'transform', 'use_hist_eq']
        conf_dict = {key:conf_dict.get(key) for key in relevant_keys}
        self.add_rate_builders(conf_dict.get('rate_builders'))
        self.transform = conf_dict.get('transform') or combine_sum
        self.use_hist_eq = conf_dict.get('use_hist_eq') or False

    def _preprocess(self):
        # Calculate dependent variables
        if self._rate_builders:
            super().with_steps_per_ms(self._rate_builders[0].steps_per_ms)
            combined_channels = set.union(*[set(rb.channels) for rb in self._rate_builders])
            super().with_channels(combined_channels)
            super().with_time_length(self._rate_builders[0].time_length)
        super()._preprocess()

    def _validate(self):
        has_common_time_length = len(set(rb.time_len_in_steps for rb in self._rate_builders)) <= 1
        assert has_common_time_length, "All constituent rate builders must have a common time length"

        has_common_steps_per_ms = len(set(rb.steps_per_ms for rb in self._rate_builders)) <= 1
        assert has_common_steps_per_ms, "All constituent rate builders must have a common step size"

    channels = get_unsettable(BaseRateBuilder, 'channels')
    steps_per_ms = get_unsettable(BaseRateBuilder, 'steps_per_ms')

    @BaseRateBuilder.time_length.setter
    @requires_rebuild
    def time_length(self, time_length_):
        if time_length_ is None:
            super().with_time_length(None)
        else:
            for rb in self._rate_builders:
                rb.time_length = time_length_

    @property
    def transform(self):
        return self._transform

    @transform.setter
    @requires_rebuild
    def transform(self, transform_):
        if transform_ is None:
            self._init_attr('_transform', combine_sum)
        else:
            self._transform = transform_


    @property
    def use_hist_eq(self):
        return self._use_hist_eq

    @use_hist_eq.setter
    @requires_rebuild
    def use_hist_eq(self, use_hist_eq_):
        self._use_hist_eq = bool(use_hist_eq_)

    @property
    def rate_builders(self):
        return_rb_list = []
        for rb in self._rate_builders:
            return_rb_list.append(rb.frozen_view())
        return return_rb_list
        
    @property_setter('rate_builders')
    def add_rate_builders(self, rate_builder_array):
        if rate_builder_array is None:
            self._init_attr('_rate_builders', [])
        else:
            for rb in rate_builder_array:
                if get_builder_type(rb) == 'rate':
                    self._rate_builders.append(rb)
                else:
                    raise TypeError("the rate_builder_array must contain only rate builder objects")
    

    def _build(self):
        # First, we build all the rate builders
        for rb in self._rate_builders:
            rb.build()

        nchannels = self._channels.size
        # Then, for each rate, builder we extend the first dimention to be equal to the number of channels with 0 as the output wherever there is no channel. If it is a zero dimensional array, we simply let it be
        channel_index_map = {self._channels[i]:i for i in range(nchannels)}
        output_rate_array_list = []
        for rb in self._rate_builders:
            if np.isscalar(rb.rate_array):
                output_rate_array_list.append(rb.rate_array)
            else:
                channel_index_list =  [channel_index_map[ch] for ch in rb._channels]
                extended_rate_array = np.zeros((nchannels,) + rb.rate_array.shape[1:])
                extended_rate_array[channel_index_list, ...] = rb.rate_array
                output_rate_array_list.append(extended_rate_array)

        final_rate_array = self._transform(output_rate_array_list)
        if self._use_hist_eq:
            # NEED TO WRITE code to implement hist eq
            raise ValueError("histogram equalization not yet implemented")
        self._rate_array = final_rate_array


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


#histogram matching
#http://stackoverflow.com/questions/32655686/histogram-matching-of-two-images-in-python-2-x (users/1461210/ali-m)
def hist_match(source, template):
    """
    Adjust the pixel values of a grayscale image such that its histogram
    matches that of a target image

    Arguments:
    -----------
        source: np.ndarray
            Image to transform; the histogram is computed over the flattened
            array
        template: np.ndarray
            Template image; can have different dimensions to source
    Returns:
    -----------
        matched: np.ndarray
            The transformed output image
    """

    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()

    # get the set of unique pixel values and their corresponding indices and
    # counts
    ###s_values, bin_idx, s_counts = np.unique(source, return_inverse=True, return_counts=True) #requires numpy 1.9
    ###t_values, t_counts = np.unique(template, return_counts=True) #requires numpy 1.9

    s_values, bin_idx = np.unique(source, return_inverse=True)
    s_counts = np.bincount(bin_idx)
    t_values, idx_tmp = np.unique(template, return_inverse=True)
    t_counts = np.bincount(idx_tmp)

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    return interp_t_values[bin_idx].reshape(oldshape)
