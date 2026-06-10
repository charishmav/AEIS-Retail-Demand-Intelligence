import joblib
import pandas as pd
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder


DATA = Path("data")
MODEL_DIR = Path("models")

MODEL_DIR.mkdir(exist_ok=True)


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

    encoders = {}

    for c in [
        "family",
        "city",
        "state",
        "type"
    ]:

        enc = LabelEncoder()

        df[c] = (
            enc.fit_transform(
                df[c]
            )
        )

        encoders[c] = enc

    return df, encoders


df = load()

df, encoders = preprocess(df)

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

X = df[features]

y = df.sales


model = XGBRegressor(

    n_estimators=324,
    max_depth=9,
    learning_rate=0.08
)

model.fit(X, y)

joblib.dump(
    model,
    "models/xgb_model.pkl"
)

joblib.dump(
    encoders,
    "models/encoders.pkl"
)

print()

print(
    "Saved model."
)