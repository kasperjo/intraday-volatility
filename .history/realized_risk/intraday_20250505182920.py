# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "arch==7.2.0",
#     "cvxcovariance==0.1.5",
#     "cvxpy==1.6.5",
#     "marimo",
#     "matplotlib==3.9.4",
#     "mosek==11.0.19",
#     "numpy==2.0.2",
#     "pandas==2.2.3",
#     "tqdm==4.67.1",
#     "xarray==2025.4.0",
# ]
# ///

import marimo

__generated_with = "0.13.0"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    cache_switch = mo.ui.switch(value=True)

    mo.vstack([
        mo.hstack([mo.md("# Enable caching"), cache_switch], justify="start", align="center"),
        mo.md("When caching is enabled, all experiment results are automatically saved to the cache/ folder. If a run with the same combination of parameters has already been cached, the saved results will be loaded instead of rerunning the experiment.")
    ])
    return (cache_switch,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Data visualization""")
    return


@app.cell
def _(five_min_returns, mo, np, plt):
    arr = (five_min_returns ** 2).mean(dim=["Date", "Asset"]).to_pandas() ** 0.5 * np.sqrt(250)
    arr.plot(marker="o")
    plt.gcf().autofmt_xdate()
    plt.xlabel("")

    mo.vstack([
        mo.md("### Annualized volatility at different time stamps"),
        plt.gcf()
    ], align="center")

    return


@app.cell
def _(intra_day_market, mo, plt):
    market = intra_day_market.squeeze()
    times = market["Time"].values

    time_dropdown = mo.ui.dropdown(
        options=times,
        value=times[0],
    )

    def make_figure(time):
        market_overnight = market.sel(Time=time)

        cov = (market * market_overnight).mean(dim="Date")
        var1 = (market * market).mean(dim="Date")
        var2 = (market_overnight * market_overnight).mean(dim="Date")

        corr = (cov / (var1 * var2) ** 0.5).to_pandas()

        fig, ax = plt.subplots()
        corr.plot(marker="o", ax=ax)
        ax.axhline(0.05, linestyle='--', label=r'$\pm 0.05$')
        ax.axhline(-0.05, linestyle='--')
        ax.axhline(corr.mean(), linestyle='--', c='r', label='mean')
        ax.legend()
        fig.autofmt_xdate()
        plt.xlabel("")
        plt.close(fig)
        return fig

    return make_figure, time_dropdown


@app.cell
def _(make_figure, mo, time_dropdown):
    mo.vstack([
        mo.hstack([mo.md("### Intraday return correlations at"), time_dropdown]),
        make_figure(time_dropdown.value)
    ], align="center")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Autocorrelation""")
    return


@app.cell
def _(mo):
    acf_toggle = mo.ui.switch(label="Run ACF experiments", value=False)

    mo.hstack([
        acf_toggle,
        mo.md("**⚠️ Note:** Takes a few minutes and is not necessary for experiments.")
    ], justify="start")

    return (acf_toggle,)


@app.cell
def _(
    acf_toggle,
    daily_log_returns,
    five_min_log_returns,
    get_acf,
    mo,
    pd,
    tqdm,
):
    if acf_toggle.value:
        daily_acf = get_acf(daily_log_returns, daily_log_returns * 0 + 1, 21, 'Date')
        intra_day_acf = {}

        for date in tqdm(five_min_log_returns['Date'].values):
            date = pd.to_datetime(date)
            _returns = five_min_log_returns.sel(Date=date)
            intra_day_acf[date] = get_acf(_returns, _returns * 0 + 1, 10, 'Time')
        _return_val = None
    else:
        _return_val = mo.vstack([mo.md("### ACF experiments have not been run.")], align="center")
    return daily_acf, intra_day_acf


@app.cell
def _(acf_toggle, intra_day_acf, mo, pd, plt):
    if acf_toggle.value:
        pd.DataFrame(intra_day_acf).T.mean(axis=0).plot(marker='o')
        plt.axhline(0.05, linestyle='--', label=r'$\pm 0.05$')
        plt.axhline(-0.05, linestyle='--')
        plt.xlabel("Number of 5-minute intervals between returns")
        plt.legend()
        plt.ylabel("Correlation")

        _return_val = mo.vstack([
            mo.md("### ACF of intra-day returns"),
            plt.gcf()
    ], align="center")
    else:
        _return_val = None
    _return_val
    return


@app.cell
def _(acf_toggle, daily_acf, mo, plt):
    if acf_toggle.value:
        daily_acf.plot(marker='o')
        plt.axhline(0.05, linestyle='--', label=r'$\pm 0.05$')
        plt.axhline(-0.05, linestyle='--')
        plt.xlabel("Number of days between returns")
        plt.ylabel("Correlation")
        plt.legend()

        _return_val = mo.vstack([
            mo.md("### ACF of daily returns"),
            plt.gcf()
    ], align="center")

    else:
        _return_val = None

    _return_val
    return


@app.cell
def _(np, xr):
    five_min_returns = xr.load_dataarray('realized_risk/data/returns_merged.nc')
    five_min_log_returns = np.log(1 + five_min_returns)
    daily_log_returns = five_min_log_returns.sum(dim='Time')
    daily_returns = np.exp(daily_log_returns) - 1
    return (
        daily_log_returns,
        daily_returns,
        five_min_log_returns,
        five_min_returns,
    )


@app.cell
def _(asset_names, daily_dropdown, daily_market, np):
    n_assets = len(asset_names)
    market_holdings = np.full(n_assets, 1 / n_assets)
    sigma_market = (daily_market ** 2).to_pandas().ewm(halflife=daily_dropdown.value).mean().squeeze() ** 0.5
    return market_holdings, sigma_market


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Covariance matrix half-lives""")
    return


@app.cell
def _(mo):
    ewm_halflife_dropdown = mo.ui.dropdown(
        options=[125, 250, 500, 1000, 2000, 5000],
        label="EWMA half-life",
        value=1000,
    )

    iewm_volahalflife_dropdown = mo.ui.dropdown(
        options=[5, 21, 63, 125, 250, 500],
        label="IEWMA volatility half-life",
        value=21,
    )
    iewm_corrhalflife_dropdown = mo.ui.dropdown(
        options=[125, 250, 500, 1000, 2000, 5000],
        label="IEWMA correlation half-life",
        value=500,
    )

    realized_halflife_dropdown = mo.ui.dropdown(
        options=[5, 21, 63, 125, 250, 500],
        label="Realized half-life",
        value=63,
    )

    realized_volahalflife_dropdown = mo.ui.dropdown(
        options=[1, 5, 10, 15, 21, 30, 40, 63, 125, 250],
        label="Realized IEWMA volatility half-life",
        value=21,
    )
    realized_corrhalflife_dropdown = mo.ui.dropdown(
        options=[5, 21, 63, 125, 250, 500],
        label="Realized IEWMA correlation half-life",
        value=63,
    )
    return (
        ewm_halflife_dropdown,
        iewm_corrhalflife_dropdown,
        iewm_volahalflife_dropdown,
        realized_corrhalflife_dropdown,
        realized_halflife_dropdown,
        realized_volahalflife_dropdown,
    )


@app.cell
def _(
    ewm_halflife_dropdown,
    iewm_corrhalflife_dropdown,
    iewm_volahalflife_dropdown,
    mo,
    realized_corrhalflife_dropdown,
    realized_halflife_dropdown,
    realized_volahalflife_dropdown,
):
    mo.vstack([
        mo.md("======================================================="),
        ewm_halflife_dropdown,
        mo.md("======================================================="),
        mo.vstack([iewm_volahalflife_dropdown, iewm_corrhalflife_dropdown], justify="start"),
        mo.md("======================================================="),
        realized_halflife_dropdown,
        mo.md("======================================================="),
        mo.vstack([realized_volahalflife_dropdown, realized_corrhalflife_dropdown], justify="start"),
        mo.md("======================================================="),
    ], align="center")
    return


@app.cell
def _(
    market_returns,
    mo,
    np,
    plt,
    risks_ewma,
    risks_iewma,
    risks_realized,
    risks_realized_iewma,
):
    risks_ewma.plot(label="EWMA")
    risks_iewma.plot(label="IEWMA")
    risks_realized.plot(label="Realized")
    risks_realized_iewma.plot(label="Realized IEWMA")
    (market_returns.rolling(window=63, center=True).std() * np.sqrt(250)).plot(label="Realized market volatiliy")
    plt.legend()

    mo.vstack([
        mo.md("### Market volatility predictions"),
        plt.gcf()
    ], align="center")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Log likelihood""")
    return


@app.cell
def _():
    ### Select half-lives (in days)
    return


@app.cell
def _(daily_returns, pd):
    to_remove = [pd.to_datetime('2015-08-21 00:00:00'), pd.to_datetime('2015-08-24 00:00:00')]
    test_dates = pd.to_datetime(daily_returns['Date'].values[500:-1])

    test_dates = list(filter(lambda x: x not in to_remove, test_dates))
    return (test_dates,)


@app.cell
def _(daily_returns, log_likelihood, pd):
    def get_log_likelihood(covariances, test_dates):
        return pd.Series([
        log_likelihood(daily_returns.shift({'Date':-1}).sel(Date=_date).values, covariances[_date].values) for _date in 
        test_dates
    ], index = test_dates)
    return (get_log_likelihood,)


@app.cell
def _(
    daily_returns,
    ewm_halflife_dropdown,
    ewma_covariance,
    get_log_likelihood,
    test_dates,
):
    ewma = {}
    _halflife = ewm_halflife_dropdown.value
    for _result in ewma_covariance(
        daily_returns,
        halflife=_halflife,
    ):
        ewma[_result.date] = _result.covariance

    ewma_likelihoods = get_log_likelihood(ewma, test_dates)
    return ewma, ewma_likelihoods


@app.cell
def _(
    daily_returns,
    get_log_likelihood,
    iewm_corrhalflife_dropdown,
    iewm_volahalflife_dropdown,
    iterated_ewma,
    market_holdings,
    scale_cov,
    sigma_market,
    test_dates,
):
    iewma = {}
    _vola_hl = iewm_volahalflife_dropdown.value
    _corr_hl = iewm_corrhalflife_dropdown.value
    for _result in iterated_ewma(
        daily_returns.to_pandas(),
        vola_halflife=_vola_hl,
        cov_halflife=_corr_hl,
    ):
        # iewma[_result.time] = _result.covariance
        iewma[_result.time] = scale_cov(_result.covariance, market_holdings, sigma_market.loc[_result.time])

    iewma_likelihoods = get_log_likelihood(iewma, test_dates)
    return iewma, iewma_likelihoods


@app.cell
def _(
    five_min_returns,
    get_log_likelihood,
    market_holdings,
    realized_ewma,
    realized_halflife_dropdown,
    scale_cov,
    sigma_market,
    test_dates,
):
    realized = {}
    _halflife = realized_halflife_dropdown.value
    for _result in realized_ewma(five_min_returns, halflife=_halflife, min_periods=1):
        # realized[_result.date] = _result.covariance
        realized[_result.date] = scale_cov(_result.covariance, market_holdings, sigma_market.loc[_result.date])

    realized_likelihoods = get_log_likelihood(realized, test_dates)
    return realized, realized_likelihoods


@app.cell
def _():
    return


@app.cell
def _(
    five_min_returns,
    get_log_likelihood,
    market_holdings,
    realized_corrhalflife_dropdown,
    realized_ewma,
    realized_volahalflife_dropdown,
    scale_cov,
    sigma_market,
    test_dates,
):
    realized_iewma = {}
    _vola_hl = realized_volahalflife_dropdown.value
    _corr_hl = realized_corrhalflife_dropdown.value
    for _result in realized_ewma(five_min_returns, halflife={'vola':_vola_hl, 'corr':_corr_hl}, min_periods=1, DCC=True, daily_scaling=True):
        # realized_iewma[_result.date] = _result.covariance
        realized_iewma[_result.date] = scale_cov(_result.covariance, market_holdings, sigma_market.loc[_result.date])

    realized_iewma_likelihoods = get_log_likelihood(realized_iewma, test_dates)
    return realized_iewma, realized_iewma_likelihoods


@app.cell
def _(
    ewma_likelihoods,
    iewma_likelihoods,
    prescient_likelihoods,
    realized_iewma_likelihoods,
    realized_likelihoods,
):
    verbose = False
    if verbose:
        print("IEWMA", iewma_likelihoods.mean() / iewma_likelihoods.mean())
        print("EWMA", ewma_likelihoods.mean() / iewma_likelihoods.mean())
        print("Realized", realized_likelihoods.mean() / iewma_likelihoods.mean())
        print("DCC", realized_iewma_likelihoods.mean() / iewma_likelihoods.mean())
        print("Prescient", prescient_likelihoods.mean() / iewma_likelihoods.mean())
    return


@app.cell
def _(five_min_returns, get_log_likelihood, pd, realized_ewma, test_dates):
    fast_covariances = {}
    asset_names = five_min_returns['Asset'].values
    for _result in realized_ewma(five_min_returns, halflife=1, min_periods=1):
        fast_covariances[_result.date] = _result.covariance

    _dates = pd.to_datetime(five_min_returns["Date"].values)
    prescient = {date_prev: fast_covariances[date_next] for date_prev, date_next in zip(_dates[:-1], _dates[1:])}

    prescient_likelihoods = get_log_likelihood(prescient, test_dates)
    return asset_names, prescient_likelihoods


@app.cell
def _(
    ewma_likelihoods,
    iewma_likelihoods,
    mo,
    plt,
    prescient_likelihoods,
    realized_iewma_likelihoods,
    realized_likelihoods,
):
    freq = 'YE'
    ewma_regret = (prescient_likelihoods - ewma_likelihoods)
    realized_regret = (prescient_likelihoods - realized_likelihoods)
    iewma_regret = (prescient_likelihoods - iewma_likelihoods)
    realized_iewma_regret = (prescient_likelihoods - realized_iewma_likelihoods)

    ewma_regret.iloc[500:].resample(freq).mean().plot(marker='o', label='EWMA')
    iewma_regret.iloc[500:].resample(freq).mean().plot(marker='o', label='IEWMA')
    realized_regret.resample(freq).mean().plot(marker='o', label='Realized')
    realized_iewma_regret.iloc[:].resample(freq).mean().plot(marker='o', label='Realized IEWMA')

    plt.legend()
    plt.gcf().autofmt_xdate()

    mo.vstack([
        mo.md("### Regret"),
        plt.gcf()
    ], align="center")
    return


@app.cell
def _(asset_names, np, pd):
    def predicted_risk(covariances, holdings, dates):
        if isinstance(holdings, np.ndarray):
            holdings = pd.DataFrame(
                np.repeat(holdings[None, :], len(dates), axis=0),
                index=dates,
                columns=asset_names
            )

        risks = pd.Series(index=dates, dtype=float)
        for date in dates:
            cov = covariances[date].values
            hold = holdings.loc[date].values

            risks.loc[date] = (hold @ cov @ hold) ** 0.5 * np.sqrt(250)

        return risks
    return (predicted_risk,)


@app.cell
def _(
    ewma,
    iewma,
    market_holdings,
    predicted_risk,
    realized,
    realized_iewma,
    test_dates,
):
    risks_ewma = predicted_risk(ewma, market_holdings, test_dates)
    risks_iewma = predicted_risk(iewma, market_holdings, test_dates)
    risks_realized = predicted_risk(realized, market_holdings, test_dates)
    risks_realized_iewma = predicted_risk(realized_iewma, market_holdings, test_dates)
    return risks_ewma, risks_iewma, risks_realized, risks_realized_iewma


@app.cell
def _(daily_returns, market_holdings, portfolio_metrics):
    market_returns = daily_returns.to_pandas() @ market_holdings
    market_metrics = portfolio_metrics(market_returns)
    return (market_returns,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Portfolios

        We construct portfolios by solving the following Markowitz mean-variance optimization problem:

        $$
        \begin{aligned}
        \text{minimize} \quad & c \\
        \text{subject to} \quad & \mathbf{1}^\top w + c = 1 \\
        & w^\top \Sigma w \leq (\sigma^{\text{tar}})^2 \\
        & \|w\|_1 \leq L^{\text{tar}} \\
        & w^{\text{min}} \leq w \leq w^{\text{max}},
        \end{aligned}
        $$

        where \( w \in \mathbf{R}^n \) denotes the portfolio weights and \( \Sigma \) is the covariance matrix of asset returns. We set the risk target \( \sigma^{\text{tar}} = 10\% \) (annualized), the leverage target \( L^{\text{tar}} = 1.6 \), and the elementwise weight bounds \( w^{\text{min}} = -15\% \) and \( w^{\text{max}} = 15\% \).
        """
    )
    return


@app.cell
def _(
    cache_backtest,
    cache_switch,
    ewm_halflife_dropdown,
    ewma,
    load_backtest,
    run_backtest,
    test_dates,
):
    if cache_switch.value:
        _config = {
            "model": "EWMA",
            "halflife": ewm_halflife_dropdown.value
                  }
        try:
            min_var_ewma = load_backtest(_config)
        except FileNotFoundError:
            min_var_ewma = run_backtest(ewma, test_dates)
            cache_backtest(min_var_ewma, _config)
    else:
        min_var_ewma = run_backtest(ewma, test_dates)
    return (min_var_ewma,)


@app.cell
def _(
    cache_backtest,
    cache_switch,
    iewm_corrhalflife_dropdown,
    iewm_volahalflife_dropdown,
    iewma,
    load_backtest,
    run_backtest,
    test_dates,
):
    if cache_switch.value:
        _config = {
            "model": "IEWMA",
            "vola_halflife": iewm_volahalflife_dropdown.value,
            "corr_halflife": iewm_corrhalflife_dropdown.value
                  }
        try:
            min_var_iewma = load_backtest(_config)
        except FileNotFoundError:
            min_var_iewma = run_backtest(iewma, test_dates)
            cache_backtest(min_var_iewma, _config)
    else:
        min_var_iewma = run_backtest(iewma, test_dates)
    return (min_var_iewma,)


@app.cell
def _(
    cache_backtest,
    cache_switch,
    load_backtest,
    realized,
    realized_halflife_dropdown,
    run_backtest,
    test_dates,
):
    if cache_switch.value:
        _config = {
            "model": "Realized",
            "halflife": realized_halflife_dropdown.value,
                  }
        try:
            min_var_realized = load_backtest(_config)
        except FileNotFoundError:
            min_var_realized = run_backtest(realized, test_dates)
            cache_backtest(min_var_realized, _config)
    else:
        min_var_realized = run_backtest(realized, test_dates)
    return (min_var_realized,)


@app.cell
def _(
    cache_backtest,
    cache_switch,
    load_backtest,
    realized_corrhalflife_dropdown,
    realized_iewma,
    realized_volahalflife_dropdown,
    run_backtest,
    test_dates,
):
    if cache_switch.value:
        _config = {
            "model": "Realized IEWMA",
            "vola_halflife": realized_volahalflife_dropdown.value,
            "corr_halflife": realized_corrhalflife_dropdown.value
                  }
        try:
            min_var_realized_iewma = load_backtest(_config)
        except FileNotFoundError:
            min_var_realized_iewma = run_backtest(realized_iewma, test_dates)
            cache_backtest(min_var_realized_iewma, _config)
    else:
        min_var_realized_iewma = run_backtest(realized_iewma, test_dates)
    return (min_var_realized_iewma,)


@app.cell
def _(
    min_var_ewma,
    min_var_iewma,
    min_var_realized,
    min_var_realized_iewma,
    mo,
    summarize_portfolio,
):
    mo.vstack([
        summarize_portfolio(min_var_ewma, label="EWMA summary"),
        summarize_portfolio(min_var_iewma, label="IEWMA summary"),
        summarize_portfolio(min_var_realized, label="Realized summary"),
        summarize_portfolio(min_var_realized_iewma, label="Realized IEWMA summary"),
    ])
    return


@app.cell
def _(min_var_iewma, min_var_realized, min_var_realized_iewma, mo, np, plt):
    def plot_sharpes(backtest, freq, label=None):
        mean = backtest.pnl.resample(freq).mean() * 250
        risk = backtest.pnl.resample(freq).std() * np.sqrt(250)
        sharpe = mean / risk

        label = f"{label} (mean {sharpe.mean():.2f}, std {sharpe.std():.2f})"
        sharpe[1:].plot(marker='o', label=label)

        return mean[:], risk[:], sharpe[:]

    freqq = "YE"
    # ewma_mean, ewma_risk, ewma_sharpe = plot_sharpes(min_var_ewma, freqq, label='EWMA')
    iewma_mean, iewma_risk, iewma_sharpe = plot_sharpes(min_var_iewma, freqq, label='IEWMA')
    realized_mean, realized_risk, realized_sharpe = plot_sharpes(min_var_realized, freqq, label='Realized')
    dcc_mean, dcc_risk, dcc_sharpe = plot_sharpes(min_var_realized_iewma, freqq, label='Realized IEWMA')
    plt.legend()

    mo.vstack([
        mo.md(f"### Realized Sharpe ratio each {freqq}"),
        plt.gcf()
    ], align="center")
    return dcc_risk, freqq, iewma_risk, realized_risk


@app.cell
def _(dcc_risk, freqq, iewma_risk, mo, plt, realized_risk):
    iewma_risk.plot(marker="o", label=f"IEWMA (mean {iewma_risk.mean():.1%}, std {iewma_risk.std():.1%})")
    realized_risk.plot(marker="o", label=f"Realized (mean {realized_risk.mean():.1%}, std {realized_risk.std():.1%})")
    dcc_risk.plot(marker="o", label=f"Realized IEWMA (mean {dcc_risk.mean():.1%}, std {dcc_risk.std():.1%})")

    plt.axhline(iewma_risk.mean(), linestyle="--")
    plt.axhline(realized_risk.mean(), linestyle="--")
    plt.axhline(dcc_risk.mean(), linestyle="--")
    plt.legend()

    mo.vstack([
        mo.md(f"### Realized risk ration each {freqq}"),
        plt.gcf()
    ], align="center")
    return


@app.cell
def _():
    # (dcc_sharpe - iewma_sharpe).plot(marker="o")
    # plt.axhline(0, linestyle="--")
    return


@app.cell
def _(
    min_var_ewma,
    min_var_iewma,
    min_var_realized,
    min_var_realized_iewma,
    mo,
    np,
    plt,
):
    halflife = 250
    (min_var_ewma.pnl.ewm(halflife=halflife, min_periods=halflife).std() * np.sqrt(250)).plot(label='EWMA')
    (min_var_iewma.pnl.ewm(halflife=halflife, min_periods=halflife).std() * np.sqrt(250)).plot(label='IEWMA')
    (min_var_realized.pnl.ewm(halflife=halflife, min_periods=halflife).std() * np.sqrt(250)).plot(label='Realized 21')
    (min_var_realized_iewma.pnl.ewm(halflife=halflife, min_periods=halflife).std() * np.sqrt(250)).plot(label='Realized IEWMA')
    # plt.ylim(0, 0.3)
    plt.legend()
    plt.gcf()

    mo.vstack([
        mo.md(f"### {halflife}-day EWMA risk"),
        plt.gcf()
    ], align="center")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Market Portfolios

        We analyze a single asset—the market—defined as the equally weighted average of all assets in our universe. The market portfolio is scaled to a 10% annualized volatility target using various volatility estimators:

        - **Daily**: Exponentially weighted volatility estimated from daily returns.
        - **Daily + Clip**: Same as *Daily*, but with returns winsorized based on predicted volatility for each date.
        - **Realized**: Exponentially weighted volatility of realized daily volatilities, where realized volatilities are computed from intraday returns.
        - **GARCH**: Volatility forecast from a GARCH(1,1) model fit to daily returns.
        """
    )
    return


@app.cell
def _(mo):
    daily_dropdown = mo.ui.dropdown(
        options=[1, 5, 10, 15, 21, 63, 125],
        label="EWMA half-life",
        value=15,
    )

    realized_dropdown = mo.ui.dropdown(
        options=[1, 5, 10, 15, 21, 63, 125],
        label="Realized half-life",
        value=15,
    )

    mo.vstack([
        mo.md("======================================================="),
        daily_dropdown,
        mo.md("======================================================="),
        realized_dropdown,
        mo.md("======================================================="),
    ], align="center")
    return daily_dropdown, realized_dropdown


@app.cell
def _(five_min_returns):
    intra_day_market = five_min_returns.mean(dim="Asset")
    intra_day_market = intra_day_market.expand_dims(Asset=["Market"])
    intra_day_market = intra_day_market.transpose("Date", "Time", "Asset").copy()
    daily_market = ((1+intra_day_market).prod(dim="Time")-1)
    return daily_market, intra_day_market


@app.cell
def _(
    daily_dropdown,
    daily_market,
    ewma_vola,
    intra_day_market,
    np,
    pd,
    realized_dropdown,
):
    volas_daily = (daily_market ** 2).to_pandas().ewm(halflife=daily_dropdown.value).mean().squeeze() ** 0.5 * np.sqrt(250)
    volas_intraday = (intra_day_market ** 2).sum(dim="Time").to_pandas().ewm(halflife=realized_dropdown.value).mean().squeeze() ** 0.5 * np.sqrt(250)

    clip_at = 3
    min_periods = 15
    volas_clip = pd.Series(index=pd.to_datetime(daily_market["Date"].values), dtype=float)
    for result in ewma_vola(daily_market.to_pandas().squeeze(),
                                       halflife=daily_dropdown.value,
                                       min_periods=min_periods, 
                                       clip_at=clip_at):
        volas_clip.loc[pd.to_datetime(result[0])] = result[1]

    volas_clip = volas_clip * np.sqrt(250)
    return volas_clip, volas_daily, volas_intraday


@app.cell
def _(mo, pd, plt, volas_clip, volas_daily, volas_garch, volas_intraday):
    plt.plot(volas_daily, label="Daily")
    plt.plot(volas_intraday, label="Intraday")
    plt.plot(volas_clip, label="Daily + Clip")
    plt.plot(volas_garch, label="GARCH")
    plt.legend()

    plt.xlim(pd.to_datetime('2006-01-01'), pd.to_datetime('2017-01-01'))

    mo.vstack([
        mo.md(f"### Risk measure based on daily returns and intarday returns"),
        plt.gcf()
    ], align="center")
    # plt.show()
    return


@app.cell
def _(volas_clip, volas_daily, volas_garch, volas_intraday):
    sigma_tar = 0.1

    holdings_daily = sigma_tar / volas_daily
    holdings_intraday = sigma_tar / volas_intraday
    holdings_clip = sigma_tar / volas_clip
    holdings_garch = sigma_tar / volas_garch
    holdings_clip.name = "Market"
    holdings_garch.name = "Market"
    return holdings_clip, holdings_daily, holdings_garch, holdings_intraday


@app.cell
def _(
    BacktestSummary,
    daily_market,
    holdings_clip,
    holdings_daily,
    holdings_garch,
    holdings_intraday,
    test_dates,
    volas_clip,
    volas_daily,
    volas_intraday,
):
    portfolio_daily = BacktestSummary(
            returns=daily_market.to_pandas().loc[test_dates],
            holdings=holdings_daily.to_frame().loc[test_dates],
            predicted_risk=(holdings_daily * volas_daily).loc[test_dates],
        )

    portfolio_intraday = BacktestSummary(
            returns=daily_market.to_pandas().loc[test_dates],
            holdings=holdings_intraday.to_frame().loc[test_dates],
            predicted_risk=(holdings_intraday * volas_intraday).loc[test_dates],
        )

    portfolio_clip = BacktestSummary(
            returns=daily_market.to_pandas().loc[test_dates],
            holdings=holdings_clip.to_frame().loc[test_dates],
            predicted_risk=(holdings_clip * volas_clip).loc[test_dates],
        )

    portfolio_garch = BacktestSummary(
            returns=daily_market.to_pandas().loc[test_dates],
            holdings=holdings_garch.to_frame().loc[test_dates],
            predicted_risk=(holdings_garch * volas_clip).loc[test_dates],
        )
    return portfolio_clip, portfolio_daily, portfolio_garch, portfolio_intraday


@app.cell
def _(
    mo,
    np,
    plt,
    portfolio_clip,
    portfolio_daily,
    portfolio_garch,
    portfolio_intraday,
):
    (portfolio_daily.pnl.ewm(halflife=125, min_periods=21).std() * np.sqrt(250)).plot(
        label=f"Daily {portfolio_daily.risk:.1%}"
    )

    (portfolio_intraday.pnl.ewm(halflife=64, min_periods=21).std() * np.sqrt(250)).plot(
        label=f"Intraday {portfolio_intraday.risk:.1%}"
    )

    (portfolio_clip.pnl.ewm(halflife=125, min_periods=21).std() * np.sqrt(250)).plot(
        label=f"Clip {portfolio_clip.risk:.1%}"
    )

    (portfolio_garch.pnl.ewm(halflife=125, min_periods=21).std() * np.sqrt(250)).plot(
        label=f"Garch {portfolio_garch.risk:.1%}"
    )
    plt.axhline(0.1, linestyle="--", label="Target")
    plt.legend()
    mo.vstack([
        mo.md("### 125-day half-life EWMA portfolio volatility"),
        plt.gcf(),
    ], align="center")
    return


@app.cell
def _(
    mo,
    portfolio_clip,
    portfolio_daily,
    portfolio_garch,
    portfolio_intraday,
    summarize_portfolio,
):
    mo.vstack([
    summarize_portfolio(portfolio_daily, "Daily summary"),
    summarize_portfolio(portfolio_clip, "Daily + Clip summary"),
    summarize_portfolio(portfolio_intraday, "Intraday summary"),
    summarize_portfolio(portfolio_garch, "GARCH summary"),
    ], align="center")

    return


@app.cell
def _(np):
    from utils.risk_model import _ewma_update

    def ewma_vola(returns, halflife=15, min_periods=1, clip_at=3):
        beta = np.exp(np.log(0.5) / halflife)

        dates = returns.index

        _date = dates[0]
        _return = returns.loc[_date]
        _ewma = _return ** 2

        yield _date, _ewma ** 0.5

        for k, _date in enumerate(dates[1:], start=1):
            t = k + 1
            _return = returns.loc[_date]

            if k >= min_periods - 1:
                _return = np.clip(_return, - clip_at * _ewma ** 0.5, clip_at * _ewma ** 0.5)

            next_val = _return ** 2
            _ewma = _ewma_update(_ewma, next_val, beta, t)

            yield _date, _ewma ** 0.5
    return (ewma_vola,)


@app.cell
def _(daily_market, np, pd, test_dates, tqdm):
    from arch import arch_model

    volas_garch = pd.Series(index=test_dates, dtype=float)
    for _date in tqdm(test_dates):
        model = arch_model(daily_market.to_pandas().squeeze().loc[:_date].iloc[:] * 1000, p=1, q=1)
        results = model.fit(disp='off')
        volas_garch.loc[_date] = (results.forecast(horizon=1).variance.values / 1000 ** 2) ** 0.5 * np.sqrt(250)
    return (volas_garch,)


@app.cell
def _(mo):
    def summarize_portfolio(portfolio, label=None):
        return mo.vstack([
            mo.md(f"### {label}" if label else ""),
            "="*147,
            portfolio.summary.round(3),
            "="*147,
    ])
    return (summarize_portfolio,)


@app.cell
def _(BacktestSummary, asset_names, daily_returns, pd, tqdm):
    def run_backtest(model, test_dates):

        from utils.min_risk import min_risk_wrapper
        from multiprocessing import Pool

        sigma_tar = 0.1

        portfolios = {}

        all_covariances = [model[_date] for _date in test_dates]
        all_dates = test_dates.copy()
        all_sigma_tars = [sigma_tar] * len(test_dates)

        all_params = list(zip(all_covariances, all_dates, all_sigma_tars))

        with Pool() as pool:
            for result in tqdm(pool.imap(min_risk_wrapper, all_params), total=len(all_params)):
                portfolios[result[0]] = result[1]

        holdings = pd.DataFrame(index=test_dates, columns=asset_names, dtype=float)
        risk = pd.Series(index=test_dates, dtype=float)

        for date in all_dates:
            holdings.loc[date] = portfolios[date].holdings
            risk.loc[date] = portfolios[date].risk

        backtest_summary = BacktestSummary(
            returns=daily_returns.to_pandas().loc[test_dates],
            holdings=holdings,
            predicted_risk=risk,
        )


        return backtest_summary
    return (run_backtest,)


@app.cell
def _():
    import marimo as mo
    import xarray as xr
    import numpy as np
    import pandas as pd
    from collections import namedtuple
    import cvxpy as cp
    from dataclasses import dataclass
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    from collections import namedtuple

    from cvx.covariance.ewma import iterated_ewma

    from utils.stats import get_acf, get_correlation
    from utils.risk_model import realized_ewma, ewma_covariance, scale_cov
    from utils.metrics import PortfolioMetrics, portfolio_metrics, BacktestSummary, log_likelihood
    return (
        BacktestSummary,
        ewma_covariance,
        get_acf,
        iterated_ewma,
        log_likelihood,
        mo,
        np,
        pd,
        plt,
        portfolio_metrics,
        realized_ewma,
        scale_cov,
        tqdm,
        xr,
    )


@app.cell
def _():
    import pickle
    import json
    import hashlib

    def _get_str_hash(config):
        json_str = json.dumps(config, sort_keys=True)
        hash = hashlib.md5(json_str.encode()).hexdigest()

        return json_str, hash

    def cache_backtest(backtest_summary, config):
        json_str, hash = _get_str_hash(config)

        with open(f"realized_risk/cache/backtests/{hash}.pkl", "wb") as f:
            pickle.dump(backtest_summary, f)

        with open(f"realized_risk/cache/backtests/config_{hash}.json", "w") as f:
            json.dump(config, f, sort_keys=True)

    def load_backtest(config):
        json_str, hash = _get_str_hash(config)

        with open(f"realized_risk/cache/backtests/{hash}.pkl", "rb") as f:
            backtest_summary = pickle.load(f)

        with open(f"realized_risk/cache/backtests/config_{hash}.json", "r") as f:
            _config = json.load(f)

        assert config == _config

        return backtest_summary

    return cache_backtest, load_backtest


if __name__ == "__main__":
    app.run()
