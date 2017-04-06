__author__ = 'Arjun'

from ratebuilder import OURateBuilder
import scipy.stats as st
import numpy as np

#########################################################################################
# IMPORTANT!! before running this test, one needs to comment out the two lines where the
# rate array is bounded in ou_rate_gen.py in function _generate_DT_OU_process
#########################################################################################

OUParams = {
    'mean': 2.0,   # [Hz]
    'sigma': 4.0,  # [Hz]
    'theta': 2.0   # [1/ms]
}

OtherParamsDistribution = {
    'steps_per_ms': 1,
    'time_length': 5000,
    'channels': range(0, 10000, 1)
}

TotalParams = OtherParamsDistribution.copy()
TotalParams.update(OUParams)

ou_gen = OURateBuilder(**TotalParams)
ou_gen.build()

# Check Distribution
mean  = OUParams['mean']
sigma = OUParams['sigma']
theta = OUParams['theta']
print("FOR DISTRIBUTION TESTING:\n")
print("Kurtosis   : {:<10.5f}Expected: {:10.5f}".format(st.kurtosis(ou_gen.rate_array.ravel(), fisher=True), 0))
print("Mean       : {:<10.5f}Expected: {:10.5f}".format(np.mean(ou_gen.rate_array.ravel()), mean))
print("Variance   : {:<10.5f}Expected: {:10.5f}".format(np.mean((ou_gen.rate_array.ravel() - mean)**2), sigma**2/(2*theta)))
print("OneStepCorr: {:<10.5f}Expected: {:10.5f}".format(
    np.mean(
        (ou_gen.rate_array[:, :-1].ravel()-mean)
        *(ou_gen.rate_array[:, 1:].ravel()-mean)), sigma**2*np.exp(-theta/ou_gen.steps_per_ms)/(2*theta)))

OtherParamsConvergence = {
    'steps_per_ms': 1,
    'time_length': 1,
    'channels': range(0, 500000, 1)
}

TotalParams = OtherParamsConvergence.copy()
TotalParams.update(OUParams)

ou_gen2 = OURateBuilder(**TotalParams)
ou_gen2.build()

print("FOR CONVERGENCE TESTING:\n")
print("Kurtosis   : {:<10.5f}Expected: {:10.5f}".format(st.kurtosis(ou_gen2.rate_array.ravel(), fisher=True), 0))
print("Mean       : {:<10.5f}Expected: {:10.5f}".format(np.mean(ou_gen2.rate_array.ravel()), mean))
print("Variance   : {:<10.5f}Expected: {:10.5f}".format(np.mean((ou_gen2.rate_array.ravel() - mean)**2), sigma**2/(2*theta)))
