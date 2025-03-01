import streamlit as st
import joblib

model = joblib.load("models/mushroom_predictor.joblib")

st.write(type(model))