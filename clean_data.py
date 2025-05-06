# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "marimo",
#     "matplotlib==3.9.4",
#     "numpy==2.0.2",
#     "openpyxl==3.1.5",
#     "pandas==2.2.3",
#     "scipy==1.13.1",
#     "xarray==2024.7.0",
# ]
# ///

import marimo

__generated_with = "0.13.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import xarray as xr
    import datetime
    import matplotlib.pyplot as plt
    return datetime, mo, np, pd, plt, xr


@app.cell
def _(datetime, intraday, pd):
    df = pd.read_excel("data/Intraday_Overnight_Stock_Returns.xlsx", skiprows=1)
    df = df.iloc[:, 1:]
    df = df.set_index(df.columns[0])
    df.index = pd.to_datetime(df.index, format="%Y%m%d")
    df.index.name = "Date"
    df.columns.name = "Asset"

    asset_names = intraday["Asset"].values
    asset_names_p1 = [a + ".1" for a in asset_names]
    asset_names_p2 = [a + ".2" for a in asset_names]

    total_intraday = df[asset_names]
    overnight = df[asset_names_p1]
    daily = df[asset_names_p2]

    ### Test data consistency ###
    pd.testing.assert_frame_equal(
        pd.DataFrame(total_intraday.values + overnight.values, index=total_intraday.index, columns=asset_names), 
        pd.DataFrame(daily.values, index=daily.index, columns=asset_names))

    pd.testing.assert_frame_equal(intraday.sum(dim="Time").to_pandas(), total_intraday, atol=1e-4)
    ##############################

    overnight = overnight.to_xarray().to_array(
        dim="Asset"
    ).expand_dims(
        Time=[datetime.time(9, 35)]
    ).transpose(
        "Date", "Time", "Asset"
    ).assign_coords(
        Asset=asset_names
    )
    return (overnight,)


@app.cell
def _(pd, xr):
    intraday = pd.read_csv("data/HF_Returns_Stocks.csv")
    intraday["Date"] = pd.to_datetime(intraday["Date"], format="%Y%m%d")
    intraday["Time"] = pd.to_datetime(intraday["Time"], format="%H:%M:%S").dt.time
    intraday = xr.Dataset.from_dataframe(intraday.set_index(["Date", "Time"]))
    intraday = intraday.to_array(dim="Asset").transpose("Date", "Time", "Asset")
    return (intraday,)


@app.cell
def _(mo):
    mo.md(r"""# Final xarray return array""")
    return


@app.cell
def _(intraday, np, overnight, xr):
    ### Save actual returns (not log returns)
    returns = (np.exp(xr.concat(
        [overnight, intraday],
        dim="Time",
        # join="inner"
    ))-1)

    ### XXX is it okay to remove this asset ??? there is one return of over -94% at 09:35 followed by 0% at 09:40, and 1300% at 09:45, which skews the volatility plot
    returns = returns.assign_coords(Time=returns["Time"].astype(str)).sel(Asset=returns["Asset"] != "XL")

    # returns.to_netcdf("data/returns_merged.nc")
    return (returns,)


@app.cell
def _(plt, returns):
    intraday_volas = (returns ** 2).mean(dim=["Asset", "Date"]).to_pandas() ** 0.5 * 250 ** 0.5
    intraday_volas.plot(marker="o")

    plt.title("Average annualized volatility")
    plt.gcf().autofmt_xdate()
    plt.gcf()
    return


if __name__ == "__main__":
    app.run()
