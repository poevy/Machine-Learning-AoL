from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "final_model.pkl"

FEATURES = [
    {
        "name": "TotalSF",
        "label": "Total SF",
        "default": 2566,
        "min": 0,
        "step": 50,
        "format": "%d",
    },
    {
        "name": "LotArea",
        "label": "Lot Area",
        "default": 8450,
        "min": 0,
        "step": 100,
        "format": "%d",
    },
    {
        "name": "YearBuilt",
        "label": "Year Built",
        "default": 2003,
        "min": 1800,
        "max": 2026,
        "step": 1,
        "format": "%d",
    },
    {
        "name": "BedroomAbvGr",
        "label": "Bedrooms",
        "default": 3,
        "min": 0,
        "step": 1,
        "format": "%d",
    },
    {
        "name": "Total_Bathrooms",
        "label": "Total Bathrooms",
        "default": 2.5,
        "min": 0.0,
        "step": 0.5,
        "format": "%.1f",
    },
    {
        "name": "GarageCars",
        "label": "Garage Cars",
        "default": 2,
        "min": 0,
        "step": 1,
        "format": "%d",
    },
]

FEATURE_NAMES = [feature["name"] for feature in FEATURES]


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def predict_price(model, values):
    input_frame = pd.DataFrame([[values[name] for name in FEATURE_NAMES]], columns=FEATURE_NAMES)
    log_prediction = model.predict(input_frame)[0]
    return float(np.expm1(log_prediction))


st.set_page_config(page_title="House Price Predictor", layout="wide")

st.markdown(
    """
    <style>
      :root {
        --bg: #f4f1ea;
        --surface: #fffdf8;
        --ink: #243236;
        --muted: #69747a;
        --line: #d9d2c4;
        --accent: #0f766e;
        --accent-dark: #0b5751;
        --field: #ffffff;
        --shadow: 0 18px 50px rgba(32, 37, 39, 0.14);
      }

      .stApp {
        background:
          linear-gradient(135deg, rgba(15, 118, 110, 0.14), transparent 34%),
          linear-gradient(315deg, rgba(151, 85, 43, 0.14), transparent 38%),
          var(--bg);
        color: var(--ink);
      }

      [data-testid="stHeader"] {
        background: transparent;
      }

      [data-testid="stToolbar"],
      [data-testid="stDecoration"],
      #MainMenu,
      footer {
        display: none;
      }

      h1, h2, h3, h4, h5, h6, p, label, span {
        color: var(--ink);
      }

      .block-container {
        max-width: 1120px;
        padding: 2rem 2rem 3rem;
      }

      .app-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 1.25rem;
        padding: 1.75rem 2rem;
        border: 1px solid var(--line);
        border-bottom: 0;
        border-radius: 8px 8px 0 0;
        background: rgba(255, 253, 248, 0.94);
        box-shadow: var(--shadow);
      }

      .eyebrow {
        margin: 0 0 0.35rem;
        color: var(--accent-dark);
        font-size: 0.78rem;
        font-weight: 750;
        letter-spacing: 0;
        text-transform: uppercase;
      }

      .app-title {
        margin: 0;
        color: var(--ink);
        font-size: clamp(1.8rem, 4vw, 3rem);
        line-height: 1.05;
        letter-spacing: 0;
      }

      .model-pill {
        display: inline-flex;
        flex: 0 0 auto;
        width: fit-content;
        border: 1px solid #d9d2c4;
        border-radius: 999px;
        padding: 0.5rem 0.85rem;
        color: #0b5751;
        background: #eef8f6;
        font-size: 0.9rem;
        font-weight: 700;
      }

      [data-testid="stForm"] {
        min-height: 27rem;
        padding: 2rem;
        border: 1px solid var(--line);
        border-radius: 0 0 0 8px;
        background: rgba(255, 253, 248, 0.94);
        box-shadow: var(--shadow);
      }

      [data-testid="stNumberInput"] label p {
        color: var(--muted);
        font-size: 0.92rem;
        font-weight: 700;
      }

      [data-testid="stNumberInput"] input {
        min-height: 48px;
        border: 1px solid var(--line);
        border-radius: 6px;
        background: var(--field);
        color: var(--ink);
        font-size: 1rem;
      }

      [data-testid="stNumberInput"] button {
        border-color: var(--line);
        background: #f7fbfa;
        color: var(--accent-dark);
      }

      [data-testid="stNumberInput"] button:hover {
        border-color: var(--accent);
        background: #eef8f6;
        color: var(--accent-dark);
      }

      [data-baseweb="input"] {
        background: var(--field);
      }

      [data-baseweb="base-input"] {
        background: var(--field);
      }

      .stButton > button {
        min-height: 48px;
        margin-top: 0.5rem;
        border: 0;
        border-radius: 6px;
        background: var(--accent);
        color: #ffffff;
        font-weight: 800;
      }

      .stButton > button:hover {
        border: 0;
        background: var(--accent-dark);
        color: #ffffff;
      }

      .result-panel {
        display: grid;
        align-content: center;
        gap: 0.75rem;
        min-height: 27rem;
        padding: 2rem;
        border: 1px solid var(--line);
        border-left: 0;
        border-radius: 0 0 8px 0;
        background:
          linear-gradient(180deg, rgba(15, 118, 110, 0.09), transparent 55%),
          #f8f3e9;
        box-shadow: var(--shadow);
      }

      .result-label {
        color: var(--muted);
        font-size: 0.9rem;
        font-weight: 800;
      }

      .result-price {
        max-width: 100%;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
        color: var(--ink);
        font-size: clamp(2rem, 4.2vw, 3.35rem);
        line-height: 1;
        font-weight: 850;
        letter-spacing: 0;
      }

      .result-price.muted {
        color: #9aa1a4;
      }

      .feature-count {
        width: fit-content;
        border-top: 1px solid var(--line);
        padding-top: 0.75rem;
        color: var(--accent-dark);
        font-size: 0.92rem;
        font-weight: 750;
      }

      @media (max-width: 820px) {
        .block-container {
          padding: 1rem;
        }

        .app-header {
          align-items: flex-start;
          flex-direction: column;
          padding: 1.5rem;
        }

        [data-testid="stForm"] {
          min-height: auto;
          padding: 1.5rem;
          border-radius: 0;
        }

        .result-panel {
          min-height: 220px;
          border-left: 1px solid var(--line);
          border-radius: 0 0 8px 8px;
          padding: 1.5rem;
        }
      }
    </style>
    """,
    unsafe_allow_html=True,
)

model = load_model()

st.markdown(
    f"""
    <header class="app-header">
      <div>
        <p class="eyebrow">Ames Housing Model</p>
        <h1 class="app-title">House Price Predictor</h1>
      </div>
      <div class="model-pill">{MODEL_PATH.name}</div>
    </header>
    """,
    unsafe_allow_html=True,
)

form_col, result_col = st.columns([0.64, 0.36], gap="large")

with form_col:
    with st.form("prediction_form"):
        values = {}
        first_row = st.columns(2)
        second_row = st.columns(2)
        third_row = st.columns(2)
        rows = [first_row, second_row, third_row]

        for index, feature in enumerate(FEATURES):
            with rows[index // 2][index % 2]:
                values[feature["name"]] = st.number_input(
                    feature["label"],
                    min_value=feature["min"],
                    max_value=feature.get("max"),
                    value=feature["default"],
                    step=feature["step"],
                    format=feature["format"],
                )

        submitted = st.form_submit_button("Predict Price", use_container_width=True)

with result_col:
    if submitted:
        price = predict_price(model, values)
        formatted_price = f"${price:,.0f}"
        price_class = "result-price"
    else:
        formatted_price = "$--"
        price_class = "result-price muted"

    st.markdown(
        f"""
        <aside class="result-panel" aria-live="polite">
          <div class="result-label">Predicted Sale Price</div>
          <div class="{price_class}">{formatted_price}</div>
          <div class="feature-count">6 model features</div>
        </aside>
        """,
        unsafe_allow_html=True,
    )
