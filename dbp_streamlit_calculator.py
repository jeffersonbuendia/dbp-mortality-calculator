# =========================================================
# DBP IN-HOSPITAL MORTALITY CALCULATOR
# Final Streamlit app using shrinkage-adjusted and recalibrated coefficients
# =========================================================

import math
import streamlit as st
import pandas as pd

# ---------------------------------------------------------
# FINAL MODEL COEFFICIENTS
# Derived from bootstrap validation, shrinkage, and intercept recalibration
# ---------------------------------------------------------
BETA_0 = 6.513771
BETA_BW = -0.006056
BETA_APGAR5 = -0.384107
BETA_EXTREME = 0.734355

# ---------------------------------------------------------
# PREDICTION FUNCTION
# ---------------------------------------------------------
def predict_risk(birth_weight_g: float, apgar5: int, extreme_prematurity: int):
    logit = (
        BETA_0
        + BETA_BW * birth_weight_g
        + BETA_APGAR5 * apgar5
        + BETA_EXTREME * extreme_prematurity
    )
    risk = 1 / (1 + math.exp(-logit))
    return logit, risk

# ---------------------------------------------------------
# RISK CATEGORY
# ---------------------------------------------------------
def risk_band(risk: float) -> str:
    if risk < 0.05:
        return "Low risk of in-hospital mortality"
    elif risk < 0.20:
        return "Intermediate risk of in-hospital mortality"
    elif risk < 0.50:
        return "High risk of in-hospital mortality"
    else:
        return "Very high risk of in-hospital mortality"

def risk_color(risk: float) -> str:
    if risk < 0.05:
        return "#2E7D32"   # green
    elif risk < 0.20:
        return "#F9A825"   # amber
    elif risk < 0.50:
        return "#EF6C00"   # orange
    else:
        return "#C62828"   # red

# ---------------------------------------------------------
# STREAMLIT PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="DBP Mortality Risk Calculator",
    page_icon="🫁",
    layout="wide"
)

# ---------------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    .main {
        background-color: #f8fafc;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    .title-box {
        background: linear-gradient(90deg, #0f172a, #1e3a8a);
        padding: 1.2rem 1.5rem;
        border-radius: 18px;
        color: white;
        margin-bottom: 1rem;
        box-shadow: 0 4px 14px rgba(0,0,0,0.08);
    }
    .subtitle-box {
        background: white;
        padding: 1rem 1.2rem;
        border-radius: 16px;
        margin-bottom: 1rem;
        border: 1px solid #e2e8f0;
    }
    .card {
        background: white;
        padding: 1.2rem;
        border-radius: 18px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 14px rgba(0,0,0,0.05);
    }
    .risk-box {
        padding: 1.2rem;
        border-radius: 18px;
        color: white;
        font-weight: 600;
        text-align: center;
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .small-note {
        font-size: 0.92rem;
        color: #475569;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------
st.markdown(
    """
    <div class="title-box">
        <h1 style="margin:0;">DBP in-hospital mortality calculator</h1>
        <p style="margin:0.35rem 0 0 0; font-size:1.02rem;">
        Clinical web calculator derived from the final shrinkage-adjusted multivariable logistic regression model
        using birth weight, Apgar score at 5 minutes, and extreme prematurity.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle-box">
        <b>Purpose.</b> This tool estimates the probability of <b>in-hospital mortality</b> in preterm infants with bronchopulmonary dysplasia.
        It is intended to support clinical judgment and bedside risk stratification, not to replace clinical assessment.
    </div>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------------
# LAYOUT
# ---------------------------------------------------------
left, right = st.columns([1, 1])

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Clinical inputs")

    with st.form("risk_form"):
        birth_weight_g = st.number_input(
            "Birth weight (g)",
            min_value=300,
            max_value=3000,
            value=800,
            step=10,
            help="Model-derived clinical range was centered around very-low-birth-weight infants."
        )

        apgar5 = st.slider(
            "Apgar score at 5 minutes",
            min_value=0,
            max_value=10,
            value=6,
            step=1
        )

        extreme_prematurity = st.radio(
            "Extreme prematurity",
            options=[0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes",
            horizontal=True
        )

        submitted = st.form_submit_button("Calculate risk")

    st.markdown(
        """
        <p class="small-note">
        Lower birth weight, lower Apgar score at 5 minutes, and extreme prematurity increase the predicted risk of in-hospital mortality.
        </p>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <p class="small-note">
        <b>Model applicability:</b> This model was derived from a cohort of infants with bronchopulmonary dysplasia and should be interpreted in that clinical context.
        </p>
        """,
        unsafe_allow_html=True
    )

    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Predicted result")

    if submitted:
        logit, risk = predict_risk(
            birth_weight_g=birth_weight_g,
            apgar5=apgar5,
            extreme_prematurity=extreme_prematurity
        )

        band = risk_band(risk)
        color = risk_color(risk)

        col1, col2 = st.columns(2)
        col1.metric("Predicted risk", f"{risk * 100:.1f}%")
        col2.metric("Risk category", band)

        st.markdown(
            f"""
            <div class="risk-box" style="background:{color};">
                {band} — estimated in-hospital mortality risk: {risk * 100:.1f}%
            </div>
            """,
            unsafe_allow_html=True
        )

        st.progress(float(min(max(risk, 0.0), 1.0)))

        st.markdown(
            """
            **Interpretation**
            - This estimate reflects baseline risk according to the final prediction model.
            - Clinical decisions should also consider factors not included in the model.
            """
        )

        with st.expander("Model details"):
            st.write(f"**Logit:** {logit:.4f}")
            st.write(f"**Predicted probability:** {risk:.4f}")
            st.latex(
                r"\text{logit}(p)=6.513771-0.006056\times \text{Birth weight}-0.384107\times \text{Apgar5}+0.734355\times \text{Extreme prematurity}"
            )
            st.markdown(
                r"""
Predicted probability is calculated as:

\[
p = \frac{1}{1 + e^{-\text{logit}(p)}}
\]
"""
            )
    else:
        st.info("Enter the clinical values and click **Calculate risk** to generate an individual prediction.")

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# EXAMPLE SCENARIOS
# ---------------------------------------------------------
st.markdown("### Example scenarios")

examples = [
    {"Birth weight (g)": 700, "Apgar 5": 5, "Extreme prematurity": 1},
    {"Birth weight (g)": 800, "Apgar 5": 6, "Extreme prematurity": 1},
    {"Birth weight (g)": 900, "Apgar 5": 7, "Extreme prematurity": 1},
    {"Birth weight (g)": 1200, "Apgar 5": 9, "Extreme prematurity": 0},
    {"Birth weight (g)": 1500, "Apgar 5": 9, "Extreme prematurity": 0},
]

for ex in examples:
    _, ex_risk = predict_risk(
        birth_weight_g=ex["Birth weight (g)"],
        apgar5=ex["Apgar 5"],
        extreme_prematurity=ex["Extreme prematurity"]
    )
    ex["Predicted risk"] = f"{ex_risk * 100:.1f}%"
    ex["Risk category"] = risk_band(ex_risk)

examples_df = pd.DataFrame(examples)
st.dataframe(examples_df, width="stretch")

# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------
st.markdown("### Clinical notes")
st.markdown(
    """
- Lower birth weight increases predicted risk.  
- Lower Apgar score at 5 minutes increases predicted risk.  
- Extreme prematurity increases predicted risk.  
- This tool should support, not replace, clinical judgment.
"""
)

st.markdown("### Citation-ready final model")
st.code(
    "logit(p) = 6.513771 - 0.006056 × Birth weight - 0.384107 × Apgar5 + 0.734355 × Extreme prematurity",
    language="text"
)
