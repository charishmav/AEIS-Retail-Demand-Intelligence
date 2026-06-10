import shap
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from xgboost import XGBRegressor
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

sample = (
    df
    .sample(
        30000,
        random_state=42
    )
)

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

X = sample[features]

y = sample.sales


model = XGBRegressor(
    n_estimators=324,
    max_depth=9,
    learning_rate=0.08
)

model.fit(X, y)

explainer = (
    shap.TreeExplainer(
        model
    )
)

values = (
    explainer.shap_values(
        X
    )
)

plt.figure()

shap.summary_plot(
    values,
    X,
    show=False
)

plt.savefig(
    "models/shap_summary.png",
    bbox_inches="tight"
)

print(
    "\nSaved → models/shap_summary.png"
)