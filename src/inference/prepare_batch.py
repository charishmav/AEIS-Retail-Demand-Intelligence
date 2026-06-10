import pandas as pd
import joblib


encoders = joblib.load(
    "models/encoders.pkl"
)


def prepare_batch(
    train,
    stores,
    transactions,
    oil
):

    train["date"] = pd.to_datetime(
        train["date"]
    )

    transactions["date"] = (
        pd.to_datetime(
            transactions["date"]
        )
    )

    oil["date"] = (
        pd.to_datetime(
            oil["date"]
        )
    )

    df = (

        train

        .merge(
            stores,
            on="store_nbr",
            how="left"
        )

        .merge(
            transactions,
            on=[
                "date",
                "store_nbr"
            ],
            how="left"
        )

        .merge(
            oil,
            on="date",
            how="left"
        )

    )

    df["transactions"] = (
        df["transactions"]
        .fillna(0)
    )
    df["dcoilwtico"] = (
    df["dcoilwtico"]
    .fillna(
        df[
            "dcoilwtico"
        ]
        .median()
    )
    )
    df["dcoilwtico"] = (

    df[
        "dcoilwtico"
    ]

    .ffill()

    .bfill()

    )

    df["month"] = (
        df.date.dt.month
    )

    df["weekday"] = (
        df.date.dt.weekday
    )

    if "family" in encoders:

        df["family"] = (

            encoders[
                "family"
            ]
            .transform(
                df["family"]
            )

        )

    return df[

        [

        "store_nbr",
        "family",
        "transactions",
        "onpromotion",
        "month",
        "weekday",
        "cluster",
        "dcoilwtico"

        ]

    ]