import streamlit as st
import pandas as pd
import joblib
import sys
import os

from pathlib import Path

ROOT = Path(__file__).parent.parent

sys.path.append(
    str(ROOT)
)

from src.inference.prepare_batch import prepare_batch
st.set_page_config(
    page_title="AEIS Retail Intelligence",
    layout="wide"
)

from src.forecasting.future_forecast import (
    generate_forecast
)
model = joblib.load(
    "models/xgb_model.pkl"
)


st.title(
    "AEIS — Retail Demand Intelligence"
)

st.subheader(
    "Predict Future Sales"
)


col1, col2 = st.columns(2)


with col1:

    store_nbr = st.number_input(
        "Store Number",
        1,
        54,
        1
    )

    family = st.number_input(
        "Family",
        0,
        40,
        2
    )

    transactions = st.number_input(
        "Transactions",
        0,
        5000,
        1500
    )

    onpromotion = st.number_input(
        "Promotion Count",
        0,
        100,
        10
    )


with col2:

    month = st.slider(
        "Month",
        1,
        12,
        6
    )

    weekday = st.slider(
        "Weekday",
        0,
        6,
        2
    )

    cluster = st.slider(
        "Cluster",
        1,
        17,
        10
    )

    dcoilwtico = st.slider(
        "Oil Price",
        20,
        120,
        60
    )


if st.button(
    "Predict"
):

    sample = pd.DataFrame([{

        "store_nbr":
        store_nbr,

        "family":
        family,

        "transactions":
        transactions,

        "onpromotion":
        onpromotion,

        "month":
        month,

        "weekday":
        weekday,

        "cluster":
        cluster,

        "dcoilwtico":
        dcoilwtico

    }])

    pred = (
        model
        .predict(
            sample
        )[0]
    )

    st.metric(
        "Predicted Sales",

        round(
            float(pred),
            2
        )
    )
st.divider()

st.subheader(
    "Batch Prediction"
)

uploaded = st.file_uploader(
    "Upload CSV",
    type=["csv"]
)

if uploaded:

    train = pd.read_csv(
        uploaded
    )

    stores = pd.read_csv(
        "data/stores.csv"
    )

    transactions = pd.read_csv(
        "data/transactions.csv"
    )

    oil = pd.read_csv(
        "data/oil.csv"
    )

    batch = prepare_batch(

        train,
        stores,
        transactions,
        oil

    )

    st.write(
        "Columns:",
        list(batch.columns)
    )

    required = [

        "store_nbr",
        "family",
        "transactions",
        "onpromotion",
        "month",
        "weekday",
        "cluster",
        "dcoilwtico"

    ]

    missing = [

        c
        for c in required
        if c not in batch.columns

    ]

    if missing:

        st.error(
            f"Missing columns: {missing}"
        )

    else:
        if st.button(
    "Run Batch Prediction"):

            st.session_state[
                "show_results"
            ] = True


        if st.session_state.get(
            "show_results",
            False):
        

            try:

                preds = model.predict(

                    batch[
                        required
                    ]

                )

                preds = [

                    max(
                        0,
                        float(p)
                    )

                    for p in preds

                ]

                batch[
                    "predicted_sales"
                ] = preds

            except Exception as e:

                st.error(
                    f"Prediction failed: {e}"
                )

                st.stop()

            st.success(
                "Prediction Completed"
            )

            st.dataframe(
                batch.head()
            )

            st.subheader(
                "Prediction Analytics"
            )

            chart = (

                batch[
                    "predicted_sales"
                ]

                .fillna(0)

                .head(100)

            )

            st.line_chart(
                chart
            )

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(

                    "Average Forecast",

                    round(
                        float(
                            batch[
                                "predicted_sales"
                            ]
                            .mean()
                        ),

                        2
                    )
                )

            with col2:

                st.metric(

                    "Median Forecast",

                    round(
                        float(
                            batch[
                                "predicted_sales"
                            ]
                            .median()
                        ),

                        2
                    )
                )

            with col3:

                st.metric(

                    "95th Percentile",

                    round(
                        float(
                            batch[
                                "predicted_sales"
                            ]
                            .quantile(
                                0.95
                            )
                        ),

                        2
                    )
                )

            st.subheader(
                "Prediction Distribution"
            )

            st.bar_chart(
                chart
            )

            csv = batch.to_csv(
                index=False
            ).encode(
                "utf-8"
            )

            st.download_button(

                "Download Results",

                csv,

                "predictions.csv",

                "text/csv"

            )
            st.divider()

            st.subheader(
                "Explainability"
            )

            path = (
                "models/shap_summary.png"
            )

            if os.path.exists(
                path
            ):

                st.image(

                    path,

                    caption=
                    "Global Feature Importance"

                )

                st.info(

                    """
            Top features influence sales prediction.

            Higher SHAP magnitude
            =
            Higher impact.
            """

                )

            else:

                st.warning(

                    "Run SHAP analysis first."

                )
            st.divider()

            st.subheader(
                "Model Comparison"
            )

            metrics = pd.read_csv(
                "models/model_metrics.csv"
            )

            st.dataframe(
                metrics
            )

            chart = (

                metrics

                .set_index(
                    "model"
                )

            )

            st.bar_chart(
                chart
            )

            best = (

                metrics

                .loc[

                    metrics[
                        "MAE"
                    ]
                    .idxmin()

                ]

            )

            st.success(

                f"""

            Best Model:
            {best["model"]}

            MAE:
            {best["MAE"]}

            """

            )