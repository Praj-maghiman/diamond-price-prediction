import streamlit as st
import joblib
import numpy as np
import pandas as pd

# 1. Load the "Brain"
model = joblib.load('diamond_model_v2.pkl')

st.title("💎 Diamond Price Predictor")
st.markdown("Enter the details of the diamond to get the estimated price")
st.markdown("Refer the certificate for the details of X,Y,Z dimensions of the diamond to be accurate")
st.markdown("---")

# 2. Create inputs for the user
carat = st.number_input("Carat Weight", min_value=0.1, max_value=5.0, value=0.7)
cut = st.selectbox("Cut Quality", ("Fair", "Good", "Very Good", "Premium", "Ideal"))
color = st.selectbox("Color", ("J", "I", "H", "G", "F", "E", "D"))
clarity = st.selectbox("Clarity", ("I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"))
depth = st.number_input("Depth", min_value=0.1, max_value=100.0, value=60.0)
table = st.number_input("Table", min_value=0.1, max_value=100.0, value=50.0)
x = st.number_input("X", min_value=0.1, max_value=100.0, value=50.0)
y = st.number_input("Y", min_value=0.1, max_value=100.0, value=50.0)
z = st.number_input("Z", min_value=0.1, max_value=100.0, value=50.0)


if st.button("Predict Price"):
    # 3. Handle the mapping (The Bridge)
    cut_mapping = {"Fair": 1, "Good": 2, "Very Good": 3, "Premium": 4, "Ideal": 5}
    cut_val = cut_mapping[cut]
    color_mapping = {"J": 1, "I": 2, "H": 3, "G": 4, "F": 5, "E": 6, "D": 7}
    color_val = color_mapping[color]
    clarity_mapping = {"I1": 1, "SI2": 2, "SI1": 3, "VS2": 4, "VS1": 5, "VVS2": 6, "VVS1": 7, "IF": 8}
    clarity_val = clarity_mapping[clarity]
    
   # 4. Prepare the data exactly like the training set
    # Updated features list (matching the new model)
    columns = ['carat', 'cut', 'color', 'clarity', 'x', 'y', 'z']
    input_df = pd.DataFrame([[carat, cut_val, color_val, clarity_val, x, y, z]],columns=columns)
 
    log_price = model.predict(input_df)
    final_price = np.expm1(log_price)[0]

    st.success(f"The estimated price is ${final_price:,.2f}")

    