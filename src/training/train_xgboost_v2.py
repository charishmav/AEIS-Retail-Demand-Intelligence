import optuna
import pandas as pd
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder

DATA = Path("data")


def load():

    train = pd.read_csv(
        DATA/"train.csv",
        parse_dates=["date"]
    )

    stores = pd.read_csv(
        DATA/"stores.csv"
    )

    oil = pd.read_csv(
        DATA/"oil.csv",
        parse_dates=["date"]
    )

    trans = pd.read_csv(
        DATA/"transactions.csv",
        parse_dates=["date"]
    )

    df = (
        train
        .merge(stores,on="store_nbr")
        .merge(
            trans,
            on=["date","store_nbr"],
            how="left"
        )
        .merge(
            oil,
            on="date",
            how="left"
        )
    )

    return df


def preprocess(df):

    df["transactions"]=(
        df["transactions"]
        .fillna(0)
    )

    df["dcoilwtico"]=(
        df["dcoilwtico"]
        .ffill()
    )

    df["month"]=df.date.dt.month
    df["weekday"]=df.date.dt.weekday

    for c in [
        "family",
        "city",
        "state",
        "type"
    ]:

        df[c]=(
            LabelEncoder()
            .fit_transform(df[c])
        )

    return df


df = preprocess(load())

train = df[
    df.date < "2017-01-01"
]

valid = df[
    df.date >= "2017-01-01"
]

features = [

"store_nbr",
"family",
"transactions",
"onpromotion",
"month",
"weekday",
"cluster",
"dcoilwtico"

]

X_train=train[features]
y_train=train.sales

X_valid=valid[features]
y_valid=valid.sales


def objective(trial):

    model = XGBRegressor(

        n_estimators=
        trial.suggest_int(
            "trees",
            100,
            400
        ),

        max_depth=
        trial.suggest_int(
            "depth",
            4,
            10
        ),

        learning_rate=
        trial.suggest_float(
            "lr",
            0.01,
            0.3
        )
    )

    model.fit(
        X_train,
        y_train
    )

    pred = (
        model.predict(
            X_valid
        )
    )

    return mean_absolute_error(
        y_valid,
        pred
    )


study = (
    optuna
    .create_study(
        direction="minimize"
    )
)

study.optimize(
    objective,
    n_trials=10
)

print()

print(
    study.best_params
)

print(
    study.best_value
)