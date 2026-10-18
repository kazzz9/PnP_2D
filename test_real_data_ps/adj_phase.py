import numpy as np
from scipy.optimize import fmin

def dephase_fun(sp_ft, phc0, phc1, ref_ph1=0):
    """
    Dephase function for NMR spectral data.

    :param sp_ft: Complex data after Fourier transform
    :param phc0: Zero order phase correction
    :param phc1: First order phase correction
    :param ref_ph1: Reference point for zero first-order phase correction
    :return: Spectral data after phase correction
    """
    n = sp_ft.shape[1]
    a_num = -np.arange(1 - ref_ph1, n - ref_ph1 + 1) / n
    sp_dp = sp_ft * np.exp(1j * np.pi / 180 * (phc0 + phc1 * a_num))
    return sp_dp

def entropy_fun(x, s):
    """
    Calculate the entropy of NMR spectral data after phase correction.

    :param x: Array containing PHC0 and PHC1
    :param s: NMR spectral data
    :return: Entropy value after phase correction and penalty for negativity
    """
    # Assume s is a 2D numpy array; flatten if it's not
    if len(s.shape) > 1:
        N, L = s.shape
    else:
        N, L = 1, s.shape[0]
        s = s.reshape(N, L)
    
    phc0 = x[0]
    phc1 = x[1]

    # Dephase using the dephase_fun defined earlier
    s0 = dephase_fun(s, phc0, phc1)
    s = np.real(s0)

    # Absolute first-order derivative
    ds_n = np.abs(np.diff(s, n=1, axis=1))

    # Normalized derivative
    p_n = ds_n / np.sum(ds_n, axis=1, keepdims=True)

    # Avoid log(0) by setting zeros to 1 (since log(1) = 0, this does not affect the sum)
    p_n[p_n == 0] = 1

    # Entropy measure
    H_n = np.sum(-p_n * np.log(p_n), axis=1)

    # Penalty for negative values to enforce positivity
    Pfun = 0.0
    as_ = s - np.abs(s)
    sumas = np.sum(as_)
    if sumas < 0:
        Pfun = Pfun + np.sum((as_ / 2) ** 2)

    # Penalty function to ensure positive bands
    P = 1000 * Pfun

    # The value of objective function
    f = np.sum(H_n) + P
    return f

def acme(NMRsp, phcf=None):
    """
    Automatic spectrum phasing function based on entropy minimization.

    :param NMRsp: NMR spectral data (complex row vector)
    :param phcf: Initial estimates of phc0 and phc1, defaults to [0, 0]
    :return: Dephased NMR spectral data, phc0, phc1
    """
    if phcf is None:
        phcf = [0, 0]
    
    # Assuming entropy_fun is defined elsewhere
    fac = fmin(entropy_fun, phcf, args=(NMRsp,))

    phc0 = fac[0]
    phc1 = fac[1]

    NMRsp_dp = dephase_fun(NMRsp, phc0, phc1)

    return NMRsp_dp, phc0, phc1
