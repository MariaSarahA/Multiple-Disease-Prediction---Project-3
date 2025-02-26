import os
import pickle
import streamlit as st
import pandas as pd
import numpy as np
from streamlit_option_menu import option_menu
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

kidney_model = pickle.load(open('kidney_model.sav', 'rb'))
liver_model = pickle.load(open('liver_model.sav', 'rb'))
parkinson_model = pickle.load(open('parkinsons_model.sav', 'rb'))


# Set background color using CSS
st.markdown("""
    <style>
        body {
            background: linear-gradient(to right, #FFDDC1, #FFEBEE, #D7FFD9, #E3F2FD);
        }
        .stApp {
            background: linear-gradient(to right, #FFDDC1, #FFEBEE, #D7FFD9, #E3F2FD);
        }
    </style>
    """, unsafe_allow_html=True)

# Sidebar Navigation
with st.sidebar:
    selected = option_menu("Multiple Disease Prediction System",
                           ["Parkinson's Prediction", "Kidney Prediction", "Liver Prediction"],
                           menu_icon='hospital-fill',
                           icons=['activity', 'heart', 'person'],
                           default_index=0)

# ---------------------------------
# 🚀 PARKINSON'S DISEASE PREDICTION
# ---------------------------------
if selected == "Parkinson's Prediction":
    st.markdown("<h1 style='color:#6A5ACD;'>🧠 Parkinson's Disease Prediction</h1>", unsafe_allow_html=True)
    
    cols = st.columns(5)  # Creating 5 columns for better alignment

    # Input fields arranged in rows within columns
    feature_names = [
        'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)',
        'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer', 'MDVP:Shimmer(dB)',
        'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR',
        'RPDE', 'DFA', 'Spread1', 'Spread2', 'D2', 'PPE'
    ]
    
    inputs = [cols[i % 5].text_input(f"🔬 {name}") for i, name in enumerate(feature_names)]
    
    if st.button("🩺 Predict Parkinson's"):
        try:
            # Check for empty inputs
            if any(x.strip() == "" for x in inputs):
                st.warning("⚠️ Please fill in all input fields before predicting.")
            else:
                user_input = [float(x) for x in inputs]  # Convert to float

                # Validate input length
                if len(user_input) != len(feature_names):
                    st.warning(f"⚠️ Expected {len(feature_names)} inputs, but received {len(user_input)}.")
                else:
                    prediction = parkinson_model.predict([user_input])
                    if prediction[0] == 1:
                        st.markdown("<h3 style='color:red;'>🟥 The person has Parkinson's disease</h3>", unsafe_allow_html=True)
                    else:
                        st.markdown("<h3 style='color:green;'>🟩 No Parkinson's disease detected</h3>", unsafe_allow_html=True)

        except ValueError:
            st.warning("⚠️ Invalid input! Please enter numerical values only.")

# ---------------------------------
# 🚰 KIDNEY DISEASE PREDICTION
# ---------------------------------
if selected == "Kidney Prediction":
    st.markdown("<h1 style='color:#20B2AA;'>🚰 Kidney Disease Prediction</h1>", unsafe_allow_html=True)
    
    cols = st.columns(5)

    inputs = [
        st.selectbox("⚧ Gender", ["Male", "Female"]),
        cols[0].text_input("🧑 Age"), cols[1].text_input("💉 Blood Pressure"), cols[2].text_input("⚗️ Specific Gravity"),
        cols[3].text_input("🧪 Albumin"), cols[4].text_input("🍬 Sugar"), cols[0].text_input("🩸 RBC Count"),
        cols[1].text_input("🦠 Pus Cells"), cols[2].text_input("🦠 Pus Cell Clumps"), cols[3].text_input("🦠 Bacteria"),
        cols[4].text_input("🩸 Blood Glucose"), cols[0].text_input("🧪 Blood Urea"), cols[1].text_input("🧪 Serum Creatinine"),
        cols[2].text_input("💦 Sodium"), cols[3].text_input("🧂 Potassium"), cols[4].text_input("☁️ Hemoglobin"),
        cols[0].text_input("🩸 Packed Cell Volume"), cols[1].text_input("🩸 White Blood Cell Count"),
        cols[2].text_input("🩸 Red Blood Cell Count"), cols[3].text_input("🦠 Hypertension"), cols[4].text_input("🩺 Diabetes"),
        cols[0].text_input("💔 Coronary Artery Disease"), cols[1].text_input("🫁 Appetite"), cols[2].text_input("🛏️ Pedal Edema"),
        cols[3].text_input("🫀 Anemia")
    ]

    if st.button("🔍 Predict Kidney Disease"):
        user_input = [float(x) for x in inputs[1:]]
        prediction = kidney_model.predict([user_input])
        result = "🟥 The person has Kidney Disease" if prediction[0] == 1 else "🟩 No Kidney Disease detected"
        st.success(result)

# ---------------------------------
# 🩺 LIVER DISEASE PREDICTION
# ---------------------------------
if selected == "Liver Prediction":
    st.markdown("<h1 style='color:#FF8C00;'>🩺 Liver Disease Prediction</h1>", unsafe_allow_html=True)
    
    cols = st.columns(5)

    inputs = [
        cols[0].text_input("🧑 Age"), 
        cols[1].text_input("🩸 Total_Bilirubin"), 
        cols[2].text_input("🩸 Direct_Bilirubin"),
        cols[3].text_input("🧪 Alkaline_Phosphatase"), 
        cols[4].text_input("🧪 Alanine_Aminotransferase"),
        cols[0].text_input("🧪 Aspartate_Aminotransferase"), 
        cols[1].text_input("🍖 Total_Proteins"),
        cols[2].text_input("🧪 Albumin"), 
        cols[3].text_input("🔬 Albumin_and_Globulin_Ratio"),
        cols[4].text_input("👤 Gender (Male=1, Female=0)")
    ]

    if st.button("🔍 Predict Liver Disease"):
        # Check if any input is empty
        if any(x == "" for x in inputs):
            st.warning("⚠️ Please fill in all fields before predicting.")
        else:
            # Convert inputs to float
            user_input = [float(x) for x in inputs]  # Include all 10 features

            # Debugging: Check feature count
            st.write(f"User Input: {user_input}")
            st.write(f"Number of Features: {len(user_input)}")

            if len(user_input) == liver_model.n_features_in_:
                prediction = liver_model.predict([user_input])
                result = "🟥 The person has Liver Disease" if prediction[0] == 1 else "🟩 No Liver Disease detected"
                st.success(result)
            else:
                st.error(f"❌ Feature count mismatch: Expected {liver_model.n_features_in_}, but got {len(user_input)}.")
