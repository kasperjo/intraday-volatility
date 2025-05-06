# Intraday Volatility

This repository contains code and data for analyzing intraday volatility. Below
are instructions for running the various components of the project.


To generate and clean the data, download the required datasets from
[here](https://mpelger.people.stanford.edu/data-and-code). Once the data is
downloa
marimo edit --sandbox clean_data.py
```

## Covariance Experiments

To run experiments related to covariance, execute the following command:

```bash
marimo edit --sandbox realized_risk/intraday.py
```

## Effective Market Time Experiments

To compute effective market time, use the following command:

```bash
marimo edit --sandbox realized_risk/effective_time.py
```

## Notes

- Ensure all dependencies are installed and the required data is downloaded before running the scripts.
- For any issues or questions, please refer to the documentation or contact the repository maintainer.ded, run the following command from the top directory:

```bash
## Data Generation and Cleaning