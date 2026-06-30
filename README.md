# AEIS — Retail Demand Intelligence Platform

## Overview

AEIS is an end-to-end AI-powered retail demand intelligence platform designed to forecast demand, generate business insights, explain model decisions, and support future planning.

The platform combines machine learning, explainability, forecasting, API serving, and interactive analytics into a production-style workflow.

---

## Features

* Retail Demand Prediction
* Batch CSV Prediction
* Automated Feature Engineering
* Explainable AI (SHAP)
* Future Forecasting (Prophet)
* Model Comparison Dashboard
* FastAPI Prediction API
* Streamlit Dashboard
* Forecast Analytics

---

## Tech Stack

### Machine Learning

* XGBoost
* Prophet
* SHAP
* Scikit-learn

### Backend

* FastAPI
* Uvicorn

### Dashboard

* Streamlit

### Data

* Pandas
* NumPy

---

## Architecture

Data Collection
↓
Preprocessing & Feature Engineering
↓
Model Training (XGBoost + Prophet)
↓
Prediction API
↓
Dashboard Analytics
↓
Forecasting
↓
Explainability

---

## Project Structure
```
AEIS/

├── api/
│   └── app.py

├── dashboard/
│   └── app.py

├── models/
│   ├── model_metrics.csv
│   └── shap_summary.png

├── src/
│   ├── preprocessing/
│   ├── training/
│   ├── forecasting/
│   ├── explainability/
│   └── inference/

├── requirements.txt

├── README.md

└── .gitignore
```
---

## Installation

Clone repository:

git clone https://github.com/charishmav/AEIS-Retail-Demand-Intelligence.git

Move into project:

cd AEIS-Retail-Demand-Intelligence

Create environment:

python -m venv venv

Activate:

venv/bin/activate

Install dependencies:

pip install -r requirements.txt

---

## Run Dashboard

streamlit run dashboard/app.py

---

## Run API

uvicorn api.app:app --reload

---


---

## Author

Charishma Arkat
