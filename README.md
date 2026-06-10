# Ames House Price Predictor

Streamlit app for predicting house sale prices from a trained XGBoost model.

## Project Structure

```text
.
├── streamlit_app.py          # Streamlit web app
├── final_model.pkl           # Best trained model used by the app
├── requirements.txt          # Python dependencies for Streamlit Cloud
├── packages.txt              # Linux package needed by XGBoost
├── .streamlit/config.toml    # Streamlit light theme config
├── data/                     # Dataset files
├── notebooks/                # Training notebook
├── models/                   # Optional retraining outputs
└── outputs/                  # Optional prediction/submission outputs
```

## Run Locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Deploy To Streamlit Cloud

Use these settings:

```text
Main file path: streamlit_app.py
```

Streamlit Cloud will install Python dependencies from `requirements.txt` and system packages from `packages.txt`.

## Model Inputs

The app uses these 6 features:

```text
TotalSF
LotArea
YearBuilt
BedroomAbvGr
Total_Bathrooms
GarageCars
```
