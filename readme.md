# Intraday Volatility

This repository contains code and data for analyzing intraday volatility. It includes tools for generating and cleaning data, computing realized covariance, and estimating effective market time using high-frequency financial data.

## 📁 Setup

### 1. Download the Data

Download the required datasets from [this Google Drive folder](https://drive.google.com/drive/folders/16RXMH26fZYagbe3pB0w_OZW2chRAhZLI?usp=sharing) and move them into the `data/` directory.  
The original data is provided by [Markus Pelger's website](https://mpelger.people.stanford.edu/data-and-code) and has been reformatted for this project.

### 2. Install Dependencies

Install the required Python packages:

```bash
pip install marimo uv
```

### 3. (Optional) Inspect Data Cleaning

To see how the raw data was cleaned and reformatted:

```bash
marimo edit --sandbox clean_data.py
```

---

## 📊 Covariance Experiments

To run experiments related to realized covariance:

```bash
marimo edit --sandbox realized_risk/intraday.py
```

## ⏱️ Effective Market Time

To compute effective market time:

```bash
marimo edit --sandbox realized_risk/effective_time.py
```

---

## 📝 Notes

- Ensure the `data/` directory is populated with the required files before running the scripts.
- Scripts are intended to be run using [Marimo](https://github.com/marimo-team/marimo), a reactive Python notebook environment.
- If you want to hide the code and only run the output, replace `edit` with `run`, e.g.:

  ```bash
  marimo run --sandbox realized_risk/intraday.py
  ```

