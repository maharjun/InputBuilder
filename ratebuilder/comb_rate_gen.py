from . import BaseRateBuilder
from genericbuilder.propdecorators import requires_built, prop_setter
from genericbuilder.tools import get_builder_type

from .combination_funcs import combine_sum
import numpy as np


class CombinedRateBuilder(BaseRateBuilder):

    built_properties = ['rate_array']

    def __init__(self, rate_builders=(), transform=combine_sum, use_hist_eq=False):
        # Initialization done directly as no None initializable property / function
        # corresponding to _rate_builders
        self._rate_builders = ()
        super().__init__()  # only purpose is to run BaseGenericBuilder init

        self.add_rate_builders(rate_builders)
        self.transform = transform
        self.use_hist_eq = use_hist_eq

    def _preprocess(self):
        # Calculate dependent variables
        if self._rate_builders:
            self._steps_per_ms = self._rate_builders[0].steps_per_ms
            combined_channels = sorted(set.union(*[set(rb.channels) for rb in self._rate_builders]))
            self._channels = np.array(combined_channels, dtype=np.uint32())
            self._channels.setflags(write=False)

    def _validate(self):
        step_length_set = set(int(rb.time_length*rb.steps_per_ms+0.5) for rb in self._rate_builders)
        has_common_time_length = len(step_length_set) <= 1
        assert has_common_time_length, "All constituent rate builders must have a common time length"

        has_common_steps_per_ms = len(set(rb.steps_per_ms for rb in self._rate_builders)) <= 1
        assert has_common_steps_per_ms, "All constituent rate builders must have a common step size"

    @property
    def transform(self):
        return self._transform

    @transform.setter
    def transform(self, transform_):
        if transform_ is None:
            self._init_attr('_transform', combine_sum)
        else:
            self._transform = transform_

    @property
    def use_hist_eq(self):
        return self._use_hist_eq

    @use_hist_eq.setter
    def use_hist_eq(self, use_hist_eq_):
        self._use_hist_eq = bool(use_hist_eq_)

    @property
    def rate_builders(self):
        return self._rate_builders

    @prop_setter
    def add_rate_builders(self, rate_builder_array):
        if rate_builder_array is None:
            self._init_attr('_rate_builders', [])
        else:
            new_rate_builders = []
            for rb in rate_builder_array:
                if get_builder_type(rb) == 'rate':
                    new_rate_builders.append(rb.copy().set_immutable())
                else:
                    raise TypeError("the rate_builder_array must contain only rate builder objects")
            self._rate_builders = self._rate_builders + tuple(new_rate_builders)

    # channels and steps_per_ms cannot be set as they are entirely derived from the constiuent
    # rate-builders
    @property
    def channels(self):
        return self._channels

    @property
    def steps_per_ms(self):
        return self._steps_per_ms

    @property
    def time_length(self):
        if self._rate_builders:
            return self._rate_builders[0].time_length
        else:
            return np.uint32(0)

    @time_length.setter
    def time_length(self, time_length_):
        new_rate_builders = []
        for rb in self._rate_builders:
            rb = rb.copy_mutable()
            rb.time_length = time_length_
            rb.set_immutable()
            new_rate_builders.append(rb)
        self._rate_builders = tuple(new_rate_builders)

    @property
    @requires_built
    def rate_array(self):
        return self._rate_array

    def _build(self):
        # First, we build all the rate builders
        new_built_rbs = []
        for rb in self._rate_builders:
            new_built_rbs.append(rb.build_copy())
        self._rate_builders = tuple(new_built_rbs)

        nchannels = self._channels.size
        # Then, for each rate-builder we extend the first dimention to be equal to the number of
        # channels with 0 as the output wherever there is no channel. If it is a zero dimensional
        # array, we simply let it be
        channel_index_map = {self._channels[i]: i for i in range(nchannels)}
        output_rate_array_list = []
        for rb in self._rate_builders:
            if np.isscalar(rb.rate_array):
                output_rate_array_list.append(rb.rate_array)
            else:
                channel_index_list = [channel_index_map[ch] for ch in rb._channels]
                extended_rate_array = np.zeros((nchannels,) + rb.rate_array.shape[1:])
                extended_rate_array[channel_index_list, ...] = rb.rate_array
                output_rate_array_list.append(extended_rate_array)

        final_rate_array = self._transform(output_rate_array_list)
        if self._use_hist_eq:
            # NEED TO WRITE code to implement hist eq
            raise ValueError("histogram equalization not yet implemented")
        self._rate_array = final_rate_array


# histogram matching
# http://stackoverflow.com/questions/32655686/histogram-matching-of-two-images-in-python-2-x (users/1461210/ali-m)
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
    # s_values, bin_idx, s_counts = np.unique(source, return_inverse=True, return_counts=True) #requires numpy 1.9
    # t_values, t_counts = np.unique(template, return_counts=True) #requires numpy 1.9

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
