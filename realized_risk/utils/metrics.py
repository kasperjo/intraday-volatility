import pandas as pd
import numpy as np
from dataclasses import dataclass
from collections import namedtuple

def log_likelihood(returns, covariance):
    n_assets = len(returns)
    inv_covariance = np.linalg.inv(covariance)

    sign, logdet = np.linalg.slogdet(covariance)

    return 0.5 * (
        -n_assets * np.log(2*np.pi) - logdet - returns @ inv_covariance @ returns
    )

PortfolioMetrics = namedtuple('PortfolioMetrics', ['mean', 'risk', 'sharpe'])
def portfolio_metrics(daily_returns):
    mean = daily_returns.mean() * 250
    risk = daily_returns.std() * np.sqrt(250)
    sharpe = mean / risk

    return PortfolioMetrics(mean, risk, sharpe)

@dataclass(frozen=True)
class BacktestSummary:
        holdings : pd.DataFrame
        returns : pd.DataFrame
        predicted_risk : pd.Series
        # predicted_pnl : pd.Series
        # predicted_factor_risk : pd.Series
        # predicted_idio_risk : pd.Series
        # status : pd.Series
        # groups : pd.Series

        def __post_init__(self):
            # Ensure that the returns and holdings are aligned
            if not self.holdings.index.equals(self.returns.index):
                raise ValueError("Holdings and returns must have the same index.")
            if not self.holdings.columns.equals(self.returns.columns):
                raise ValueError("Holdings and returns must have the same columns.") 

        @property
        def dates(self):
            return self.holdings.index

        @property
        def pnl(self):
            return (self.holdings.shift(1) * self.returns).sum(axis=1)

        @property
        def portfolio_value(self):
            return self.pnl.cumsum()
        
        @property
        def mean_predicted_risk(self):
            return (self.predicted_risk ** 2).mean() ** 0.5

        @property
        def mean_pnl(self):
            return self.pnl.mean() * 250  # annualize

        @property
        def risk(self):
            return self.pnl.std() * np.sqrt(250)

        @property
        def sharpe(self):
            return self.mean_pnl / self.risk if self.risk > 0 else np.nan

        @property
        def summary(self):
            return pd.Series(
                [self.mean_pnl,
                 self.risk,
                 self.sharpe,
                 self.mean_predicted_risk,
                 self.mean_GMV,
                 self.mean_TMV,
                 self.mean_exposure,
                 self.mean_pnl / self.mean_GMV,
                ],
                index=[
                    'mean_pnl',
                    'risk',
                    'sharpe',
                    'mean_predicted_risk',
                    'mean_GMV',
                    'mean_TMV',
                    'mean_exposure',
                    'mean_pnl / mean_GMV',
                      ] 
            )

        @property
        def GMV(self):
            return self.holdings.abs().sum(axis=1)

        @property
        def mean_GMV(self):
            return self.GMV.mean()

        @property 
        def trades(self):
            return self.holdings.diff(axis=0)

        @property
        def TMV(self):
            return self.trades.abs().sum(axis=1)

        @property
        def exposure(self):
            return self.holdings.sum(axis=1)

        @property
        def mean_exposure(self):
            return self.exposure.mean()

        @property
        def mean_TMV(self):
            return self.TMV.mean()
