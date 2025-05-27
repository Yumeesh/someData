import pandas as pd
from functools import lru_cache
from datetime import datetime

FILTER_COLUMNS = [
    "Product",
    "Customer category",
    "Customer",
    "MJPRDesc",
    "Region",
    "Sales office",
    "Sales Head"
]

@lru_cache(maxsize=1)
def load_data():
    df = pd.read_csv("synthetic_sales_data.csv")
    # Standardize column names
    df = df.rename(columns={
        "Customer Category": "Customer category",
        "SalesOffice": "Sales office",
        "Price": "y",
        "Year": "year",
        "Month": "month"
    })
    # Combine year and month into a date (ds)
    df["ds"] = df.apply(lambda row: pd.to_datetime(f"{row['year']} {row['month']} 1"), axis=1)
    return df

def get_unique_filters():
    df = load_data()
    unique_filters = {}
    for col in FILTER_COLUMNS:
        unique_filters[col] = sorted(df[col].dropna().unique().tolist())
    return unique_filters

def filter_sales_data(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    # Apply dynamic filters (supporting multiple selections)
    for col, values in (filters or {}).items():
        if col in df.columns and values:
            df = df[df[col].isin(values)]
    # Aggregate by ds and sum y
    agg_df = df.groupby("ds", as_index=False)["y"].sum()
    return agg_df


def forecast_sales(df: pd.DataFrame, periods: int = 30) -> pd.DataFrame:
    from prophet import Prophet
    # Prophet expects columns: ds (datetime), y (float)
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(periods)


import matplotlib.pyplot as plt
import base64
import io
import os

def prophet_plot(df: pd.DataFrame, periods: int = 30) -> str:
    from prophet import Prophet
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    fig = model.plot(forecast)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=60)
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    # Save to file for /plots endpoint
    os.makedirs("plots", exist_ok=True)
    fname = f"plots/plot_{hash(str(df.head()))}_{periods}.b64"
    with open(fname, "w") as f:
        f.write(b64)
    return b64


def prophet_components(df: pd.DataFrame, periods: int = 30) -> str:
    from prophet import Prophet
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    fig = model.plot_components(forecast)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=60)
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    # Save to file for /plots endpoint
    os.makedirs("plots", exist_ok=True)
    fname = f"plots/components_{hash(str(df.head()))}_{periods}.b64"
    with open(fname, "w") as f:
        f.write(b64)
    return b64
