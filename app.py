# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 00:07:54 2025

@author: Lenovo
"""

import streamlit as st
import pickle
import numpy as np

# Load models and encoders
import joblib
model_college = joblib.load('model_college_compressed.pkl')


with open('model_branch.pkl', 'rb') as f:
    model_branch = pickle.load(f)

with open('label_enc_college.pkl', 'rb') as f:
    label_enc_college = pickle.load(f)

with open('label_enc_branch.pkl', 'rb') as f:
    label_enc_branch = pickle.load(f)

# Define categories
categories = ['OPEN', 'EWS', 'SC', 'SEBC', 'ST', 'EX']

st.title("🎓 College & Branch Predictor")
st.write("Enter your rank and category to predict a suitable college and branch.")

# User inputs
merit_rank = st.number_input("Enter your Merit Rank", min_value=1, value=1000)
category = st.selectbox("Select Your Category", categories)

if st.button("Predict"):
    # Prepare input
    input_data = [[merit_rank if cat == category else 999999 for cat in categories]]

    # Predict
    pred_college = model_college.predict(input_data)
    pred_branch = model_branch.predict(input_data)

    # Decode predictions
    predicted_college = label_enc_college.inverse_transform(pred_college)[0]
    predicted_branch = label_enc_branch.inverse_transform(pred_branch)[0]

    # Display results
    st.success(f"🏫 Predicted College: **{predicted_college}**")
    st.success(f"📘 Predicted Branch: **{predicted_branch}**")
