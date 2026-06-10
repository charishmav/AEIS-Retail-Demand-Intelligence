import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")


def load():

    train = pd.read_csv(
        DATA_DIR / "train.csv",
        parse_dates=["date"]
    )

    stores = pd.read_csv(
        DATA_DIR / "stores.csv"
    )

    transactions = pd.read_csv(
        DATA_DIR / "transactions.csv",
        parse_dates=["date"]
    )

    oil = pd.read_csv(
        DATA_DIR / "oil.csv",
        parse_dates=["date"]
    )

    df = (
        train
        .merge(stores, on="store_nbr", how="left")
        .merge(transactions,
               on=["date", "store_nbr"],
               how="left")
        .merge(oil,
               on="date",
               how="left")
    )

    return df


def build_features(df):

    df["dcoilwtico"] = (
        df["dcoilwtico"]
        .ffill()
    )

    df["transactions"] = (
        df["transactions"]
        .fillna(0)
    )

    df["year"] = df.date.dt.year
    df["month"] = df.date.dt.month
    df["weekday"] = df.date.dt.weekday
    df["week"] = df.date.dt.isocalendar().week

    df = df.sort_values(
        ["store_nbr", "family", "date"]
    )

    df["lag_7"] = (
        df.groupby(
            ["store_nbr", "family"]
        )["sales"]
        .shift(7)
    )

    df["lag_14"] = (
        df.groupby(
            ["store_nbr", "family"]
        )["sales"]
        .shift(14)
    )

    df["rolling_7"] = (
        df.groupby(
            ["store_nbr", "family"]
        )["sales"]
        .transform(
            lambda x:
            x.shift(1)
             .rolling(7)
             .mean()
        )
    )

    df["rolling_30"] = (
        df.groupby(
            ["store_nbr", "family"]
        )["sales"]
        .transform(
            lambda x:
            x.shift(1)
             .rolling(30)
             .mean()
        )
    )

    df.fillna(0, inplace=True)

    return df


if __name__ == "__main__":

    df = load()

    df = build_features(df)

    print("\nFinal Shape:")
    print(df.shape)

    print("\nNew Features:")

    print([
        "lag_7",
        "lag_14",
        "rolling_7",
        "rolling_30"
    ])

    print(df.head())