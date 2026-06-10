import pandas as pd
from prophet import Prophet
from pathlib import Path


DATA = Path(
    "data"
)


def generate_forecast():

    train = pd.read_csv(

        DATA /
        "train.csv",

        parse_dates=[
            "date"
        ]

    )

    daily = (

        train

        .groupby(
            "date"
        )

        [
            "sales"
        ]

        .mean()

        .reset_index()

    )

    daily.columns = [

        "ds",
        "y"

    ]

    model = Prophet(

        yearly_seasonality=True,

        weekly_seasonality=True,

        daily_seasonality=False

    )

    model.fit(
        daily
    )

    future = (

        model

        .make_future_dataframe(

            periods=30

        )

    )

    forecast = (

        model

        .predict(
            future
        )

    )

    return (

        forecast[

            [

            "ds",

            "yhat"

            ]

        ]

        .tail(
            30
        )

        .rename(

            columns={

                "ds":
                "date",

                "yhat":
                "forecast"

            }

        )

    )