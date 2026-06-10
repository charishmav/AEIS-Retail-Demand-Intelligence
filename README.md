# AEIS — Retail Demand Intelligence Platform

## Overview

AEIS is an end-to-end AI-powered retail demand intelligence platform designed to forecast sales, generate business insights, explain predictions, and support future planning.

The system combines machine learning, explainable AI, forecasting, API serving, and interactive dashboards into a production-style workflow.

---

## Features

* Retail Demand Prediction
* Batch CSV Prediction
* Feature Engineering Pipeline
* Explainable AI (SHAP)
* Future Forecasting (Prophet)
* Model Comparison Dashboard
* FastAPI Prediction API
* Streamlit Analytics Dashboard

---

## Tech Stack

* Python
* XGBoost
* Prophet
* SHAP
* FastAPI
* Streamlit
* Pandas
* Scikit-learn

---

## Architecture

Data
→ Feature Engineering
→ XGBoost Training
→ Prediction API
→ Dashboard
→ Forecasting
→ Explainability

---

## Project Structure

AEIS/
├── dashboard/
├── api/
├── src/
├── models/
├── requirements.txt

---

## Run Locally

Install:

pip install -r requirements.txt

Run Dashboard:

streamlit run dashboard/app.py

Run API:

uvicorn api.app:app --reload

---

## Future Improvements

* Multi-dataset support
* Automated schema detection
* LSTM forecasting
* Cloud deployment
* Monitoring

---

## Author

Charishma Arkat
