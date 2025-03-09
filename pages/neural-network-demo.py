import streamlit as st
import pandas as pd
from keras import models

from utils import get_nn_scaler_and_encode, nn_encode

model = models.load_model("models/oral_cancer_predictor.keras")

scaler, data_encoder = get_nn_scaler_and_encode("datasets/oral_cancer.csv")

st.title("Neural Network Demo")
st.subheader("Oral Cancer Prediction")

st.write(":red[**DISCLAIMER:** Due to a bad dataset, which lowers the importance of risk factors (which actually is crucial in real-world), this model doesn't perform well (~50% accuracy).]")

user_input = {}

user_input["Age"] = st.number_input("Age", 15, 101)

for topic in data_encoder:
    if (topic == "Oral Cancer (Diagnosis)"):
        continue

    user_input[topic] = st.selectbox(topic, data_encoder[topic])

@st.dialog("You...")
def predict():
    df_user_input = pd.DataFrame(user_input, columns=user_input.keys(), index=[0])[scaler.feature_names_in_]

    try:
        df_user_input, _ = nn_encode(df_user_input)
    except:
        pass

    scaled_input = scaler.transform(df_user_input)

    df_user_input = pd.DataFrame(scaled_input, columns=df_user_input.columns, index=[0])[scaler.feature_names_in_]

    result = model.predict(df_user_input).round()

    print(result)

    st.write(":red[have Oral Cancer]" if result == 0
              else ":green[don't have Oral Cancer]")

st.button("Predict", on_click=predict)