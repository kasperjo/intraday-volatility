

import marimo

__generated_with = "0.13.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import xarray as xr
    import numpy as np
    import pandas as pd
    from collections import namedtuple
    import matplotlib.pyplot as plt
    return mo, np, plt, xr


@app.cell
def _(five_min_returns, mo, np, plt):
    arr = (five_min_returns ** 2).mean(dim=["Date", "Asset"]).to_pandas() ** 0.5 * np.sqrt(250)
    arr.plot(marker="o")
    plt.gcf().autofmt_xdate()
    plt.xlabel("")

    mo.vstack([
        mo.md("# Annualized volatility at different time stamps"),
        plt.gcf()
    ], align="center")
    return


@app.cell
def _():
    hours_per_day = 24
    hours_open = 6.5
    hours_closed = hours_per_day-6.5
    return hours_closed, hours_open, hours_per_day


@app.cell
def _(xr):
    five_min_returns = xr.load_dataarray('realized_risk/data/returns_merged.nc')

    overnight = five_min_returns.sel(Time="09:35:00")
    intraday = (1+five_min_returns.sel(Time= five_min_returns.Time >= "09:40:00")).prod(dim="Time") - 1
    total = (1+five_min_returns).prod(dim="Time") - 1
    return five_min_returns, intraday, overnight, total


@app.cell
def _(intraday, overnight, total):
    var_overnight = overnight.var(dim="Date") * 250 ** 2
    var_intraday = intraday.var(dim="Date") * 250 ** 2
    var_total = total.var(dim="Date") * 250 ** 2

    print(f"mean overnight variance: {var_overnight.mean().values:.0f}")
    print(f"mean intraday variance: {var_intraday.mean().values:.0f}")
    print(f"mean total variance: {var_total.mean().values:.0f}")

    print(f"mean overnight / mean total: {(var_overnight.mean() / var_total.mean()).values:.2f}")
    return var_overnight, var_total


@app.cell
def _(mo):
    mo.md(r""" """)
    return


@app.cell
def _(hours_closed, hours_per_day, mo, plt, var_overnight, var_total):
    ratio = (var_overnight / var_total).values
    plt.hist(ratio, bins=50)
    plt.axvline(ratio.mean(), c="r", linestyle="--", linewidth=3, label=f"mean {ratio.mean():.2f}")
    plt.axvline(hours_closed / hours_per_day, c="white", linestyle="--", linewidth=3, label="theoretical")
    plt.legend()

    mo.vstack([
        mo.md("# Ratio of overnight return to total return for 330 assets"),
        plt.gcf()
    ], align="center")
    return


@app.cell
def _(hours_closed, hours_open, mo, plt, var_overnight, var_total):
    x = (var_overnight / var_total).values

    effective_hours_closed =  x / (1-x) * hours_open
    slowdown = effective_hours_closed / hours_closed 

    plt.hist(slowdown, bins=50)
    plt.axvline(slowdown.mean(), c="r", linestyle="--", linewidth=3, label=f"mean {slowdown.mean():.2f}")
    plt.xlabel("Hours")
    plt.legend()

    mo.vstack([
        mo.md("# Effective length of one hour for the 330 assets"),
        plt.gcf()
    ], align="center")
    return


if __name__ == "__main__":
    app.run()
