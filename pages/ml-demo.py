import streamlit as st
import pandas as pd
import joblib

from utils import get_scaler_and_encode, label_set, kebab_to_heading

model = joblib.load("models/mushroom_predictor.joblib")
scaler, global_data_encode = get_scaler_and_encode("datasets/mushrooms.csv")

data_encode = global_data_encode.get_data_encode()
del data_encode["class"]

user_input = {}

@st.dialog("Your Mushroom is...")
def predict():
    df_user_input = pd.DataFrame(user_input, columns=user_input.keys(), index=[0])

    for column in df_user_input.columns:
        df_user_input[column].loc[0] = global_data_encode.encode(
            column,
            df_user_input[column].loc[0]
        )

    df_user_input = scaler.transform(df_user_input)

    edible = model.predict(df_user_input)[0]

    st.write(f"**{":green[Edible]" if edible else ":red[Poisonous]"}**")

st.title("ML Demo")
st.subheader("Edible Mushroom")

for column in data_encode:
    def get_label(value):
        return label_set[column][value]
    
    user_input[column] = st.selectbox(kebab_to_heading(column), options=data_encode[column], format_func=get_label)

st.button(label="Predict", on_click=predict)