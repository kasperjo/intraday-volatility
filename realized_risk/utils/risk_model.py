from collections import namedtuple
import numpy as np
import pandas as pd

def scale_cov(covariance, holdings_market, sigma_market, true_scale=True):
    """
    Scales the covariance matrix so that the predicted market volatility matches sigma_m.

    Parameters
    ----------
    covariance : pd.DataFrame
        Covariance matrix of asset returns.
    holdings_market : np.ndarray
        Market portfolio holdings.
    sigma_market : float
        Target market volatility.

    Returns
    -------
    np.ndarray
        Scaled covariance matrix.
    """
    asset_names = covariance.columns
    n_assets = len(asset_names)
    covariance = covariance.values

    sigma_model = np.sqrt(holdings_market @ covariance @ holdings_market)

    scale_factor = (sigma_market ** 2 - sigma_model ** 2) / sigma_model ** 4
    direction = covariance @ holdings_market

    
    if not true_scale: ### XXX for some weird reason works better...???
        scale_factor = sigma_market ** 2 / sigma_model ** 2
        return pd.DataFrame(scale_factor * covariance,
                        index=asset_names,
                        columns=asset_names)
    else:        
        return pd.DataFrame(covariance + scale_factor * np.outer(direction, direction),
                            index=asset_names,
                            columns=asset_names)

def _scale_matrix(matrix, scale):
    return scale[:, None] * matrix * scale[None, :]

def _ewma_update(ewma, next_val, beta, t):
    return (beta - beta ** t) / (1 - beta ** t) *  ewma + (1 - beta) / (1 - beta ** t) * next_val

def _combine(correlation, volas):
    return _scale_matrix(
        _scale_matrix(correlation, 1 / np.diag(correlation) ** 0.5), volas
    )

EWMA = namedtuple('EWMA', ['date', 'covariance'])
def ewma_covariance(daily_returns, halflife, min_periods=1):

    def f(k):
        if k < min_periods - 1:
            return EWMA(_date, None)
        else:
            return EWMA(_date, pd.DataFrame(_ewma, index=asset_names, columns=asset_names))

    asset_names = daily_returns['Asset'].values
    dates = pd.to_datetime(daily_returns['Date'].values)

    _date = dates[0]
    _returns = daily_returns.sel(Date=_date).values
    _ewma = np.outer(_returns, _returns)
    beta = np.exp(np.log(0.5) / halflife)

    yield f(k=0)

    for k, _date in enumerate(dates[1:], start=1):
        t = k + 1
        _returns = daily_returns.sel(Date=_date).values
        next_val = np.outer(_returns, _returns)
        _ewma = _ewma_update(_ewma, next_val, beta, t)
        yield f(k=k)

def _daily_var(returns, max_lag=None):
    """
    Parameters
    ----------
    returns : np.ndarray
        m x n matrix of returns (m = n_times, n = n_assets)
    max_lag : int

    Returns
    -------
    float  
        Daily variance, accounting for correlation between overnight return and intra-day returns
    """    
    max_lag = max_lag or len(returns)

    var = (returns ** 2).sum(axis=0)
    # for k in range(1, max_lag):
        # var += 2 * returns[0] * returns[k]
        # var += 2 * (returns[:-k] * returns[k:]).sum(axis=0)

    return var

def _daily_cov(returns, max_lag=None):
    """
    Parameters
    ----------
    returns : np.ndarray
        m x n matrix of returns (m = n_times, n = n_assets)
    max_lag : int

    Returns
    -------
    float  
        Daily covariance, accounting for correlation between overnight return and intra-day returns
    """
    max_lag = max_lag or len(returns)

    cov = returns.T @ returns

    # for k in range(1, max_lag):
        # cov += 2 * (returns[:-k].T @ returns[k:])
        # cov += 2 * np.outer(returns[0], returns[k])

    # return (cov + cov.T) / 2
    return cov



    


RealizedEwma = namedtuple('RealizedEwma', ['date', 'covariance'])
def realized_ewma(intra_day_returns, halflife, min_periods=1, DCC=False, **kwargs):

    daily_scaling = kwargs.get('daily_scaling', False)

    def _realized_covariance(intra_day_returns, DCC=False):
        nonlocal _ewma_var

        for date in pd.to_datetime(intra_day_returns['Date'].values):
            returns = intra_day_returns.sel(Date=date).values

            if DCC:
                # # vars = (np.prod(1 + returns, axis=0) - 1) ** 2
                # vars = (returns ** 2).sum(axis=0) # XXX
                # adj_returns = returns / _ewma_var[None, :] ** 0.5
                # # adj_returns = (returns / vars[None, :] ** 0.5)#[1:] # XXX vars or _ewma_vars???
                # correlation = adj_returns.T @ adj_returns

                vars = _daily_var(returns)
                # adj_returns = returns / vars[None, :] ** 0.5
                
                if daily_scaling:
                    adj_returns = returns / vars[None, :] ** 0.5
                else:
                    adj_returns = returns / _ewma_var[None, :] ** 0.5 # XXX vars or _ewma_vars???
                correlation = _daily_cov(adj_returns)

                yield date, (vars, correlation)

            else:
                # yield date, returns.T @ returns
                yield date, _daily_cov(returns)

    asset_names = intra_day_returns['Asset'].values
    dates = pd.to_datetime(intra_day_returns['Date'].values)

    def f(k):
        if k < min_periods - 1:
            return RealizedEwma(_date, None)
        else:
            return RealizedEwma(_date, pd.DataFrame(_ewma, index=asset_names, columns=asset_names))

    _date = dates[0]
    _returns = intra_day_returns.sel(Date=_date).values

    if DCC:
        halflife_vola = halflife['vola']
        halflife_corr = halflife['corr']
        beta_vola = np.exp(np.log(0.5) / halflife_vola)
        beta_corr = np.exp(np.log(0.5) / halflife_corr)

        # _ewma_var = (np.prod(1 + _returns, axis=0) - 1) ** 2
        # _ewma_var = (_returns ** 2).sum(axis=0) # XXX
        # _adj_returns = (_returns / _ewma_var ** 0.5)#[1:]
        # _ewma_corr = _adj_returns.T @ _adj_returns
        # _ewma = _combine(_ewma_corr, _ewma_var ** 0.5)

        _ewma_var = _daily_var(_returns)
        _adj_returns = _returns / _ewma_var[None, :] ** 0.5
        _ewma_corr = _daily_cov(_adj_returns)
        _ewma = _combine(_ewma_corr, _ewma_var ** 0.5)

    else:
        beta = np.exp(np.log(0.5) / halflife)
        # _ewma_var = None
        # _ewma = _returns.T @ _returns

        _ewma_var = None
        _ewma = _daily_cov(_returns)

    yield f(k=0)

    for k, (_date, next_val) in enumerate(_realized_covariance(intra_day_returns.sel(Date=dates[1:]), DCC), start=1):

        t = k + 1

        if DCC:
            next_var, next_corr = next_val[0], next_val[1]
            _ewma_var = _ewma_update(_ewma_var, next_var, beta_vola, t)
            _ewma_corr = _ewma_update(_ewma_corr, next_corr, beta_corr, t)
            _ewma = _combine(_ewma_corr, _ewma_var ** 0.5)
        else:
            _ewma = _ewma_update(_ewma, next_val, beta, t)

        yield f(k=k)