import pandas as pd
from pathlib import Path
from xgboost import XGBRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)

from sklearn.preprocessing import LabelEncoder


DATA = Path("data")


def load():

    train = pd.read_csv(
        DATA / "train.csv",
        parse_dates=["date"]
    )

    stores = pd.read_csv(
        DATA / "stores.csv"
    )

    transactions = pd.read_csv(
        DATA / "transactions.csv",
        parse_dates=["date"]
    )

    oil = pd.read_csv(
        DATA / "oil.csv",
        parse_dates=["date"]
    )

    df = (
        train
        .merge(stores,
               on="store_nbr")
        .merge(
            transactions,
            on=["date", "store_nbr"],
            how="left"
        )
        .merge(
            oil,
            on="date",
            how="left"
        )
    )

    return df


def feature_engineering(df):

    df["transactions"] = (
        df["transactions"]
        .fillna(0)
    )

    df["dcoilwtico"] = (
        df["dcoilwtico"]
        .ffill()
    )

    df["month"] = (
        df.date.dt.month
    )

    df["weekday"] = (
        df.date.dt.weekday
    )

    df["week"] = (
        df.date.dt.isocalendar().week
    )

    for c in [
        "family",
        "city",
        "state",
        "type"
    ]:

        enc = LabelEncoder()

        df[c] = (
            enc
            .fit_transform(
                df[c]
            )
        )

    return df


df = load()

df = feature_engineering(df)

cutoff = "2017-01-01"

train = (
    df[
        df.date
        < cutoff
    ]
)

valid = (
    df[
        df.date
        >= cutoff
    ]
)

features = [

    "store_nbr",
    "family",
    "onpromotion",
    "transactions",
    "cluster",
    "month",
    "weekday",
    "week",
    "dcoilwtico"

]

X_train = train[features]
y_train = train["sales"]

X_valid = valid[features]
y_valid = valid["sales"]


model = XGBRegressor(
    n_estimators=200,
    max_depth=8,
    learning_rate=0.05
)

model.fit(
    X_train,
    y_train
)

pred = model.predict(
    X_valid
)

mae = mean_absolute_error(
    y_valid,
    pred
)

rmse = (
    mean_squared_error(
        y_valid,
        pred
    ) ** 0.5
)

print()

print("MAE:", mae)

print("RMSE:", rmse)