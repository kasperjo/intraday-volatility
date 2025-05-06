import numpy as np
import pandas as pd
import xarray as xr

def get_correlation(arr1, arr2, weights):
    """
    Parameters
    ----------
    arr1 : xr.DataArray
    arr2 : xr.DataArray
    weights : xr.DataArray

    Returns
    -------
    float
    weighted correlation between arr1 and arr2
    """

    if type(arr1) == xr.DataArray:
        cov = (arr1 * arr2 * weights).sum(dim='Asset') / weights.sum(dim='Asset')
        var1 = (arr1 * arr1 * weights).sum(dim='Asset') / weights.sum(dim='Asset')
        var2 = (arr2 * arr2 * weights).sum(dim='Asset') / weights.sum(dim='Asset')
    elif type(arr1) == pd.DataFrame:
        cov = (arr1 * arr2 * weights).sum(axis=1) / weights.sum(axis=1)
        var1 = (arr1 * arr1 * weights).sum(axis=1) / weights.sum(axis=1)
        var2 = (arr2 * arr2 * weights).sum(axis=1) / weights.sum(axis=1)

    ### XXX: how to do this???
    # return cov.mean() / (var1.mean() * var2.mean()) ** 0.5
    return (cov / (var1 * var2) ** 0.5).mean()

def get_acf(arr, weights, n_steps, dim='Date'):
    """
    Parameters
    ----------
    arr : xr.DataArray
    weights : xr.DataArray

    Returns
    -------
    float
    weighted correlation between arr1 and arr2
    """

    acf = []
    for i in range(n_steps):
        if type(arr) == xr.DataArray:
            acf.append(get_correlation(arr, arr.shift({dim : -i}), weights))
        elif type(arr) == pd.DataFrame:
            acf.append(get_correlation(arr, arr.shift(-i), weights.values))
    return pd.Series(acf).astype(float)



def ar(halflife, sigmas, n_samples, rng):
    """
    Parameters
    ----------
    halflife : float
        ar-model half-life
    sigmas : xr.DataArray
        DataArray of asset volatilities; sigmas.dims = ['Data', 'Asset']
    n_samples
        number of samples
    rng : np.random._generator.Generator
        random number generator

    Returns
    -------
    np.ndarray
        ar models for each asset
    """
    if halflife is None:
        return sigmas * rng.normal(size=sigmas.shape)

    n_assets = len(sigmas['Asset'])

    ### ar-coefficient
    a = np.exp(np.log(1/2) / halflife) 
    sigma_eps = np.sqrt(1 - a ** 2) # variance of eps in ar-model; to make noise (z) variance 1

    ### Initialize array
    arr = np.zeros(shape=sigmas.shape)
    arr[0] = np.random.normal(loc=0, scale=1, size=(n_assets,))

    for t in range(1,len(arr)):
        eps_t = np.random.normal(loc=0, scale=sigma_eps, size=(n_assets,))
        arr[t] = a * arr[t-1] + eps_t

    return arr