import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")


def load_data():

    train = pd.read_csv(
        DATA_DIR / "train.csv",
        parse_dates=["date"]
    )

    test = pd.read_csv(
        DATA_DIR / "test.csv",
        parse_dates=["date"]
    )

    stores = pd.read_csv(
        DATA_DIR / "stores.csv"
    )

    oil = pd.read_csv(
        DATA_DIR / "oil.csv",
        parse_dates=["date"]
    )

    holidays = pd.read_csv(
        DATA_DIR / "holidays_events.csv",
        parse_dates=["date"]
    )

    transactions = pd.read_csv(
        DATA_DIR / "transactions.csv",
        parse_dates=["date"]
    )

    return (
        train,
        test,
        stores,
        oil,
        holidays,
        transactions
    )


def preprocess():

    (
        train,
        test,
        stores,
        oil,
        holidays,
        transactions
    ) = load_data()

    train = train.merge(
        stores,
        on="store_nbr",
        how="left"
    )

    train = train.merge(
        transactions,
        on=[
            "date",
            "store_nbr"
        ],
        how="left"
    )

    train = train.merge(
        oil,
        on="date",
        how="left"
    )

    train["dcoilwtico"] = (
        train["dcoilwtico"]
        .ffill()
    )

    train["year"] = train["date"].dt.year
    train["month"] = train["date"].dt.month
    train["day"] = train["date"].dt.day
    train["weekday"] = (
        train["date"]
        .dt.weekday
    )

    train.fillna(0, inplace=True)

    print("\nData Shape:")
    print(train.shape)

    print("\nColumns:")
    print(train.columns)

    print("\nPreview:")
    print(train.head())

    return train


if __name__ == "__main__":
    preprocess()