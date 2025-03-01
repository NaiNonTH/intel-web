import streamlit as st
import pandas as pd

st.title("ML Introduction")

st.write("""
         ##### Table of Contents
         * [Edible Mushroom](#edible-mushroom)
           * [Preparing Data](#preparing-data)
           * [Model Theorem](#model-theorem)
         """)

st.write("""
         ### Edible Mushroom
         Imagine that you're lost in a random jungle, starving, and suddenly, you found a mushroom but you weren't sure if the mushroom you got is poisonous or not.
         
         This **Machine Learning** model will predict whether the mushroom is edible or not according to its properties, like how they look and how they smell.

         #### Preparing Data

         First of all, I needed to find a dataset to train my AI model and I came across with [this one](https://www.kaggle.com/datasets/uciml/mushroom-classification). It contains 23 columns or 22 features with one label: `class`, containing two values: `p` for poisonous and `e` for edible—not poisonous.

         Once I got my dataset, I loaded it as a dataframe `df` and began summarizing the dataset by using this code:

         ```py
         df.describe()
         ```

         And here's what I got after running it:
         """)

from PIL import Image as img

st.image(img.open("images/ml-describe.png"))

st.write("""
         You'll be able to tell that all values inside the dataset are single characters. So I knew that I needed to encode all of them into a number before using them for training.

         But before doing that, I had to make sure that there are no duplicated data or missing values.

         When I checked for any missing values...

         ```py
         df.isna().any()
         ```
         """)

st.image(img.open("images/ml-null-check.png"))

st.write("""
         They all turned out to have none of them. And when I checked for any duplicates...

         ```py
         bool(df.duplicated().any())
         ```

         ...it also returned `False`, which means there were no duplicates either. That's a pretty good sign for a good dataset.

         But either way, I still had to encode dataset values so let's do it.

         At first, I was thinking of writing a dictionary containings a list for each columns containing a new value that are going to replace the existing one. But writing a dictionary for all 23 columns by hand, which I think they're all important, isn't going to be a good idea.

         So, I decided to loop through each column of the dataset, extract all possible values, and write them into a dictionary.

         ```py
         data_encode = {}

         for column in df.columns:
             data_encode[column] = df[column].unique().tolist()
         ```
         
         Then, I wrote an encoder maker function that returned a function that returned a value in the dictionary according to the column given when making it.

         ```py
         def make_data_encoder(column_name):
             def data_encoder(value):
                 return data_encode[column_name].index(value)
             return data_encoder
         ```

         This will return a number according to the index of that character in the dictionary.

         Now, I was ready to encode all values in the dataset.

         ```py
         for column in df.columns:
             data_encoder = make_data_encoder(column)
             df[column] = df[column].map(data_encoder)
         ```

         Alright, let's see how's our dataset.
         """)

st.image(img.open("images/ml-after-encode.png"))

st.write("""
         Nice! Now, they're all a number.

         After all of that, I separated a `class` label from the rest of the features that I need.

         ```py
         from sklearn.preprocessing import MinMaxScaler
         
         scaler = MinMaxScaler()
         
         y = df["class"]
         x = df.drop(columns=["class"])
         x = pd.DataFrame(scaler.fit_transform(x), columns=x.columns)
         ```

         As you can see, I also scaled the feature values in the `x` into numbers from 0 to 1.

         And finally, I splitted them into a training dataset and testing dataset.

         ```py
         from sklearn.model_selection import train_test_split
         
         x_train, x_test, y_train, y_test = train_test_split(x, y)
         ```

         And that's pretty much it for preparing the data before using them for training.
         """)