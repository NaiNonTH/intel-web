import streamlit as st
from PIL import Image

st.title("Neural Network Introduction")

st.write("""
         ##### Table of Contents
         * [Oral Cancer Predictor](#oral-cancer-predictor)
           * [Preparing Data](#preparing-data)
           * [Models Used and Theorem](#models-used-and-theorem)
           * [Modeling](#modeling)
           * [Fixing the Model](#fixing-the-model)
           * [Try Demo](#try-demo)
         """)

st.write("""
         ### Oral Cancer Predictor

         Using the known data according to the dataset to model a neural network AI model that predicts if the patient has Oral Cancer.

         #### Preparing Data

         I found this [dataset](https://www.kaggle.com/datasets/ankushpanday2/oral-cancer-prediction-dataset) that contains preservable data required to predict if the patient has Oral Cancer. It has 25 columns with one being a label: `Oral Cancer (Diagnosis)`
         """)

with st.expander("The other 24 features are:"):
    st.write("""
             - `ID`: Unique identifier assigned to each patient record for tracking purposes
             - `Country`: Patient's country of residence, which may reveal geographic patterns in oral cancer prevalence
             - `Age`: Patient's age in years, a significant factor as oral cancer risk increases with age
             - `Gender`: Patient's gender
               - Male
               - Female
             - `Tobacco Use`: Indicates whether the patient uses tobacco products (smoking, chewing), a major risk factor
               - Yes
               - No
             - `Alcohol Consumption`: Documents alcohol use, which can significantly increase oral cancer risk
               - Yes
               - No
             - `HPV Infection`: Human Papillomavirus infection status, an emerging risk factor especially for oropharyngeal cancers
               - Yes
               - No
             - `Betel Quid Use`: Records use of betel quid/paan, a known carcinogen common in South and Southeast Asia
               - Yes
               - No
             - `Chronic Sun Exposure`: Extended sun exposure is linked to lip cancers and other oral malignancies
               - Yes
               - No
             - `Poor Oral Hygiene`: Inadequate dental care may contribute to chronic inflammation and increased cancer risk
               - Yes
               - No
             - `Diet (Fruits & Vegetables Intake)`: Nutritional status, as diets rich in fruits and vegetables may have protective effects
               - Low
               - Moderate
               - High
             - `Family History of Cancer`: Indicates genetic predisposition to cancer
               - Yes
               - No
             - `Compromised Immune System`: Immune status, as immunocompromised individuals face higher cancer risks
               - Yes
               - No
             - `Oral Lesions`: Presence of abnormal tissue in the mouth that may indicate precancerous or cancerous conditions
               - Yes
               - No
             - `Unexplained Bleeding`: Spontaneous bleeding in the oral cavity, a potential warning sign
               - Yes
               - No
             - `Difficulty Swallowing`: Dysphagia may indicate tumors affecting the throat or esophagus
               - Yes
               - No
             - `White or Red Patches in Mouth`: Presence of leukoplakia (white) or erythroplakia (red) patches, which may be precancerous
               - Yes
               - No
             - `Tumor Size (cm)`: Numerical measurement of tumor diameter, an important prognostic factor
             - `Cancer Stage`: TNM staging system indicating cancer progression and spread
               - 0 (No Cancer)
               - 1
               - 2
               - 3
               - 4
             - `Treatment Type`: Primary treatment approach used
               - Surgery
               - Radiation
               - Chemotherapy
               - Targeted Therapy
               - No Treatment
             - `Survival Rate (5-Year, %)`: Percentage of patients surviving five years after diagnosis, a key outcome measure
             - `Cost of Treatment (USD)`: Total medical expenses in US dollars, reflecting economic impact on healthcare systems
             - `Economic Burden (Lost Workdays per Year)`: Productivity loss measured by workdays missed due to illness and treatment
             - `Early Diagnosis`: Whether the cancer was detected at an early stage, significantly affecting prognosis
               - Yes
               - No
             - `Oral Cancer (Diagnosis)`: The target variable indicating whether the patient has been diagnosed with oral cancer
               - Yes
               - No
             """)
    
st.write("""
         Like when I did in training machine learning model, I loaded the dataset (with the `ID` being removed) as `df` and checked for duplicated data and missing values.

         ```py
         # check for missing values
         df.isna().any()
         ```
         """)

st.image(Image.open("images/nn-duplicates.png"))

st.write("""
         Although there were no missing values, my dataset without the `ID` *has* duplicates, so I removed them.

         ```py
         df = df.drop_duplicates()
         ```

         Alright. Now, there should be no duplicates. So I starting encoding categorical data, turning them into a number.

         ```py
         df["Country"], country_list = df["Country"].factorize()

         df["Gender"] = df["Gender"].map({
             "Female": 1,
             "Male": 0
         })

         df["Treatment Type"] = df["Treatment Type"].map({
             "Surgery": 4,
             "Radiation": 3,
             "Chemotherapy": 2,
             "Targeted Therapy": 1,
             "No Treatment": 0
         })
         
         df["Diet (Fruits & Vegetables Intake)"] = df["Diet (Fruits & Vegetables Intake)"].map({
             "Low": 0,
             "Moderate": 1,
             "High": 2
         })

         # combine all topics that contains Yes/No values into a list
         # and loop over it for a cleaner code
         yes_no_list = [
             "Tobacco Use",
             "Alcohol Consumption",
             "HPV Infection",
             "Betel Quid Use",
             "Chronic Sun Exposure",
             "Poor Oral Hygiene",
             "Family History of Cancer",
             "Compromised Immune System",
             "Oral Lesions",
             "Unexplained Bleeding",
             "Difficulty Swallowing",
             "White or Red Patches in Mouth",
             "Early Diagnosis",
             "Oral Cancer (Diagnosis)"
         ]
         
         for column in yes_no_list:
             df[column] = df[column].map({ "Yes": 1, "No": 0 })
         ```

         Okay, let's see how's our dataset now.
         """)

st.image(Image.open("images/nn-after-encode.png"))

st.write("""
         They're all numbers now. Looks good!

         But out of my curiosity, I tried making a heatmap out of them to see the correlations between each data.

         ```py
         import matplotlib.pyplot as plt
         import seaborn as sns
         
         corr = df.corr(method="spearman")
         
         plt.figure(figsize=(12, 10))
         sns.heatmap(corr, annot=True, fmt=".1f", cmap="vanimo")
         plt.show()
         ```

         As you can see, I also use `seaborn` library to help me easily plot the `matplotlib` heatmap. And here's the result.         
         """)

st.image(Image.open("images/nn-heatmap.png"))

st.write("""
         What's interesting to me is that the our dataset shows that there are many features—most of them are pre-diagnosis data—aren't correlated to each other that much, especially with the label: `Oral Cancer (Diagnosis)`. This somehow doesn't make sense; the tobacco use, the alcohol, HPV infection, sun exposure, and chewing tobaccos are one of the risk factors of Oral Cancer, and they shouldn't be this far apart. [This guy](https://www.kaggle.com/code/muhammadaashirirshad/oral-cancer-analysis-the-truth#6.-Conclusions-and-Key-Findings) also used this datasets to create a machine learning model, but with the post-diagnosis data removed, the model doesn't perform well.

         We'll stick with this data for now, and I will remove the post-diagnosis data later. I then scaled the dataset using the `MinMaxScaler`, just like what I did in my machine learning model training.

         ```py
         from sklearn.preprocessing import MinMaxScaler

         scaler = MinMaxScaler()
         
         y = df["Oral Cancer (Diagnosis)"]
         x = df.drop(columns=["Oral Cancer (Diagnosis)"])
         x = pd.DataFrame(scaler.fit_transform(x), columns=x.columns)
         ```

         Now all of the data should've been correctly separated as `x` and `y` and scaled. Now I can just split both of them into a training and a testing dataset.

         ```py
         from sklearn.model_selection import train_test_split
         
         x_train, x_test, y_train, y_test = train_test_split(x, y)
         ```

         And that's pretty much it for preparing the data before using them for training my neural network model.

         #### Model Theorem

         Since the data isn't going to be quite large, I decided to use a regular **Feed Forward** algorithm.

         ##### Feed Forward

         Source: https://deepai.org/machine-learning-glossary-and-terms/feed-forward-neural-network

         Feed Forward is one of the simplest neural network model. It contains three layers:
         * **Input Layer** consists of neurons that receives inputs and pass them to the next layer.
         * **Hidden Layers** is considered a computational engine of neural network. Each hidden layer's neurons take the weighted sum of the outputs from the previous layer, apply an activation function to allow the model to learn complex patterns, and pass the result to the next layer.
         * **Output Layer** is the final layer where outputs for the given inputs are produced.

         The input flows forward from layer to layer until the prediction is made; we call this a **Feedforward Phase**. After that, the error is calculated and propagates back through the network, adjusting the weights to minimize the error; we call this a **Backpropagation Phase**.

         However, it comes with some challenges. Adjusting the hidden layer including the number of neurons can be cumbersome but it significantly affects the model performance. Overfitting is also a common issue with Feed Forward, where the model only performs well on the training data and not the unseen data.

         #### Modeling

         The feed forward model that I set up is pretty simple. It only consists of an input layer, a big hidden layer containing 8 neurons with `relu` activation function, and an output layer using `sigmoid` activation that is suitable for binary classification.

         ```py
         from keras import models, layers

         model = models.Sequential()
         
         model.add(layers.Input(shape=(x_train.shape[1],)))
         model.add(layers.Dense(8, activation="relu"))
         model.add(layers.Dense(1, activation="sigmoid"))
         ```

         And then, I compiled the model using `adam` optimizer.

         ```py
         model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
         ```

         Finally, I train and evaluate the data.
         """)

st.image("images/nn-evaluation.png")

st.write("""
         Obviously, the model would still perform well. This is actually because I included the post-diagnosis model, which isn't really what you would need in real-world practice.

         Now, I'm going to fix the model by removing post-diagnosis data.

         #### Fixing the Model

         This time, I'm going to remove all post-diagnosis data before splitting the dataset and use it for modeling again.

         ```py
         from sklearn.preprocessing import MinMaxScaler
         
         scaler = MinMaxScaler()
         
         x = df.drop(columns=([
             "Oral Cancer (Diagnosis)", 
             "Tumor Size (cm)",
             "Cancer Stage",
             "Treatment Type",
             "Survival Rate (5-Year, %)",
             "Cost of Treatment (USD)",
             "Economic Burden (Lost Workdays per Year)"
         ]))
         x = pd.DataFrame(scaler.fit_transform(x), columns=x.columns)
         ```

         Since the code for the rest are the same, I'm not going to show them again. Here's the result after making our model useful for real-world usage.
         """)

st.image("images/nn-evaluation2.png")

st.write("""
         You'll see that the performace of the model heavily tanked, thanks to our bad dataset that doesn't present real-world situation well.

         I'll be honest, when I made a heatmap, I felt skeptical about the dataset. Like, why are the data correlation only grouped to one specific part of the features. So I did more research about this dataset and I found that guy I mentioned who made a model out of it and addressed this issue. So I took him as my inspiration in making this model.

         #### Try Demo

         Disclaimer that the model in the demo is the one without the post-diagnosis data, so the model *WILL* perform pretty badly.
         """)

st.page_link("pages/neural-network-demo.py", label="Try Demo")