from fastapi import FastAPI
import joblib
import pandas as pd


app = FastAPI(
    title="AEIS Retail Forecast API"
)

model = joblib.load(
    "models/xgb_model.pkl"
)


FEATURES = [

"store_nbr",
"family",
"transactions",
"onpromotion",
"month",
"weekday",
"cluster",
"dcoilwtico"

]


@app.get("/health")
def health():

    return {
        "status": "running"
    }

@app.post("/predict")
def predict(payload: dict):

    df = pd.DataFrame([payload])

    print("\nINPUT:")
    print(df)

    pred = model.predict(df)

    print("\nRAW PRED:")
    print(pred)

    return {
        "predicted_sales": float(pred[0])
    }
