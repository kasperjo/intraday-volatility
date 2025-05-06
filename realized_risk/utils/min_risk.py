from collections import namedtuple
import numpy as np
import pandas as pd
import cvxpy as cp

Portfolio = namedtuple('Portfolio', ['holdings', 'risk', 'exposure'])
def min_risk_portfolio(covariance, sigma_tar):
    import warnings
    warnings.filterwarnings('ignore')

    n_assets = len(covariance)
    asset_names = covariance.columns
    covariance = covariance.values

    holdings = cp.Variable(n_assets)
    var = cp.quad_form(holdings, covariance, assume_PSD=True) * 250
    objective = cp.Maximize(holdings.sum())
    constraints = [var <= sigma_tar ** 2, holdings >= -0.15, holdings <= 0.15, cp.norm(holdings, 1) <= 1.6]
    # constraints = [var <= sigma_tar ** 2, holdings>=0]

    problem = cp.Problem(objective, constraints)
    problem.solve(solver='MOSEK')

    return Portfolio(
        holdings=pd.Series(holdings.value, index=asset_names), 
        risk=np.sqrt(var.value), 
        exposure=holdings.value.sum()
                    )

def min_risk_wrapper(params):
    covariance, date, sigma_tar = params
    return date, min_risk_portfolio(covariance, sigma_tar)