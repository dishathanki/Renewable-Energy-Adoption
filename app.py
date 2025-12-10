import streamlit as st
import numpy as np
import joblib
import os

# -------------------- CONFIG --------------------
MODEL_FILENAME = "D:\EDUNET\ENERGY_ADOPTION\Renewable_Energy_Adoption_model.pkl"
CLASS_LABELS = {0: "No Adoption", 1: "Adoption"}

# -------------------- PAGE SETUP --------------------
st.set_page_config(page_title="Renewable Adoption Predictor", page_icon="⚡🌿", layout="centered")
st.markdown("""
<style>
.main { padding: 20px; }
.result-card { padding: 20px; border-radius: 12px; background-color: #f6fdf7;
               border: 1px solid #c8e6c9; box-shadow: 0px 2px 8px rgba(0,0,0,0.05); }
.result-card-warn { padding: 20px; border-radius: 12px; background-color: #fff8e1;
                    border: 1px solid #ffe082; box-shadow: 0px 2px 8px rgba(0,0,0,0.05); }
.input-card { padding: 20px 25px; border-radius: 12px; background-color: #ffffff;
              border: 1px solid #e0e0e0; box-shadow: 0px 1px 6px rgba(0,0,0,0.05); }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center;'>⚡ Renewable Energy Adoption Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size:17px;'>Enter 4 numeric inputs and get a simple Adoption / No Adoption output.</p>", unsafe_allow_html=True)
st.write("")

# -------------------- LOAD MODEL --------------------
if not os.path.exists(MODEL_FILENAME):
    st.error(f"Model file `{MODEL_FILENAME}` not found.")
    st.stop()

try:
    model = joblib.load(MODEL_FILENAME)
except Exception as e:
    st.error("Failed to load model.")
    st.write(e)
    st.stop()

# -------------------- INPUTS --------------------
st.markdown("<div class='input-card'>", unsafe_allow_html=True)
st.subheader("🔧 Input Parameters")

col1, col2 = st.columns(2)
with col1:
    carbon_emissions = st.number_input("Carbon Emissions", value=0.0, format="%.6f")
with col2:
    energy_output = st.number_input("Energy Output", value=0.0, format="%.6f")

col3, col4 = st.columns(2)
with col3:
    renewability_index = st.number_input("Renewability Index", value=0.0, format="%.6f")
with col4:
    cost_efficiency = st.number_input("Cost Efficiency", value=0.0, format="%.6f")

st.markdown("</div>", unsafe_allow_html=True)
st.write("")

# -------------------- PREDICT --------------------
center = st.columns(3)[1]
with center:
    predict_btn = st.button("🔍 Predict", use_container_width=True)

if predict_btn:
    try:
        X = np.array([[float(carbon_emissions), 
                       float(energy_output),
                       float(renewability_index),
                       float(cost_efficiency)]])

        pred = model.predict(X)
        pred_value = int(pred[0])

        label_text = CLASS_LABELS.get(pred_value, str(pred_value))

        # Show only final prediction card — no probability
        st.write("")
        if pred_value == 1:
            st.markdown("<div class='result-card'>", unsafe_allow_html=True)
            st.markdown(f"<h3>✅ {label_text}</h3>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='result-card-warn'>", unsafe_allow_html=True)
            st.markdown(f"<h3>⚠️ {label_text}</h3>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    except Exception as e:
        st.error("Prediction failed.")
        import traceback
        st.text(traceback.format_exc())

# -------------------- FOOTER --------------------
st.markdown("""
<hr>
<p style='text-align:center; color:gray; font-size:14px;'>
Built with ❤️ using Streamlit
</p>
""", unsafe_allow_html=True)
