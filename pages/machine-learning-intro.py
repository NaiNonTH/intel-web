import streamlit as st

st.title("ML Introduction")

st.write("""
         ##### Table of Contents
         * [Edible Mushroom](#edible-mushroom)
           * [Preparing Data](#preparing-data)
           * [Models Used and Theorem](#models-used-and-theorem)
           * [Modeling](#modeling)
           * [Try Demo](#try-demo)
         """)

st.write("""
         ### Edible Mushroom
         Imagine that you're lost in a random jungle, starving, and suddenly, you found a mushroom but you weren't sure if the mushroom you got is poisonous or not.
         
         This **Machine Learning** model will predict whether the mushroom is edible or not according to its properties, like how they look and how they smell.

         #### Preparing Data

         First of all, I needed to find a dataset to train my AI model and I came across with [this one](https://www.kaggle.com/datasets/uciml/mushroom-classification). It contains 23 columns or 22 features with one label: `class`, containing two values: `p` for poisonous and `e` for edible—not poisonous.
         """)

with st.expander("And the rest 22 features are:"):
    st.write("""
         * `cap-shape`: The overall form of the mushroom's cap (pileus).
           * bell = `b`
           * conical = `c`
           * convex = `x`
           * flat = `f`
           * knobbed = `k`
           * sunken = `s`
         * `cap-surface`: The texture of the cap's upper surface.
           * fibrous = `f`
           * grooves = `g`
           * scaly = `y`
           * smooth = `s` 
         * `cap-color`:  The coloration of the mushroom's cap - varies widely between species and can change with age or damage.
           * brown = `n`
           * buff = `b`
           * cinnamon = `c`
           * gray=g,green = `r`
           * pink = `p`
           * purple = `u`
           * red = `e`
           * white = `w`
           * yellow = `y`
         * `bruises`: Indicates whether the mushroom flesh bruises or changes color when touched or damaged
           * bruises = `t`
           * no = `f`
         * `odor`: Describes the smell of the mushroom, which can be a critical identification feature
           * almond = `a`
           * anise = `l`
           * creosote = `c`
           * fishy = `y`
           * foul = `f`
           * musty = `m`
           * none = `n`
           * pungent = `p`
           * spicy = `s`
         * `gill-attachment`: Describes how the gills attach to the stalk, an important taxonomic characteristic
           * attached = `a`
           * descending = `d`
           * free = `f`
           * notched = `n`
         * `gill-spacing`: Indicates how closely the gills are positioned to each other
           * close = `c`
           * crowded = `w`
           * distant = `d`
         * `gill-size`: Describes the width of the gills
           * broad = `b`
           * narrow = `n`
         * `gill-color`: The color of the gills, which can change with mushroom age and is important for identification
           * black = `k`
           * brown = `n`
           * buff = `b`
           * chocolate = `h`
           * gray = `g`
           * green = `r`
           * orange = `o`
           * pink = `p`
           * purple = `u`
           * red = `e`
           * white = `w`
           * yellow = `y`
         * `stalk-shape`: Describes how the stalk's diameter changes from top to bottom
           * enlarging = `e`
           * tapering = `t`
         * `stalk-root`: Describes the shape and characteristics of the base of the stalk
           * bulbous = `b`
           * club = `c`
           * cup = `u`
           * equal = `e`
           * rhizomorphs = `z`
           * rooted = `r`
           * missing = `?`
         * `stalk-surface-above-ring`: Texture of the stalk surface above the ring (if present)
           * fibrous = `f`
           * scaly = `y`
           * silky = `k`
           * smooth = `s`
         * `stalk-surface-below-ring`: Texture of the stalk surface below the ring (if present)
           * fibrous = `f`
           * scaly = `y`
           * silky = `k`
           * smooth = `s`
         * `stalk-color-above-ring`: Color of the stalk above the ring (if present)
           * brown = `n`
           * buff = `b`
           * cinnamon = `c`
           * gray = `g`
           * orange = `o`
           * pink = `p`
           * red = `e`
           * white = `w`
           * yellow = `y`
         * `stalk-color-below-ring`: Color of the stalk below the ring (if present)
           * brown = `n`
           * buff = `b`
           * cinnamon = `c`
           * gray = `g`
           * orange = `o`
           * pink = `p`
           * red = `e`
           * white = `w`
           * yellow = `y`
         * `veil-type`: Describes the type of veil, a membrane that protects the developing gills
           * partial = `p`
           * universal = `u`
         * `veil-color`: Color of the veil
           * brown = `n`
           * orange = `o`
           * white = `w`
           * yellow = `y`
         * `ring-number`: Indicates how many rings are on the stalk, formed from the partial veil
           * none = `n`
           * one = `o`
           * two = `t`
         * `ring-type`: Describes the shape and characteristics of the ring(s) on the stalk
           * cobwebby = `c`
           * evanescent = `e`
           * flaring = `f`
           * large = `l`
           * none = `n`
           * pendant = `p`
           * sheathing = `s`
           * zone = `z`
         * `spore-print-color`: Color of the spore print, a critical feature for mushroom identification
           * black = `k`
           * brown = `n`
           * buff = `b`
           * chocolate = `h`
           * green = `r`
           * orange = `o`
           * purple = `u`
           * white = `w`
           * yellow = `y`
         * `population`: Describes how the mushrooms grow in relation to each other
           * abundant = `a`
           * clustered = `c`
           * numerous = `n`
           * scattered = `s`
           * several = `v`
           * solitary = `y`
         * `habitat`: The environment where the mushroom typically grows
           * grasses = `g`
           * leaves = `l`
           * meadows = `m`
           * paths = `p`
           * urban = `u`
           * waste = `w`
           * woods = `d`""")

st.write("""
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

         ...it also returned `False`, which means there were no duplicates either. That's a pretty good sign for a good dataset (or so I thought, because there *will* be more issues).

         But either way, I still had to encode dataset values so let's do it.

         At first, I was thinking of writing a dictionary containings a list for each columns containing a new value that are going to replace the existing one. But writing a dictionary for all 23 columns by hand, which I *"thought"* they're all important, isn't going to be a good idea.

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

         But wait a minute, let me just print the dictionary out...

         ```py
         {'class': ['p', 'e'],
          'cap-shape': ['x', 'b', 's', 'f', 'k', 'c'],
          'cap-surface': ['s', 'y', 'f', 'g'],
          'cap-color': ['n', 'y', 'w', 'g', 'e', 'p', 'b', 'u', 'c', 'r'],
          'bruises': ['t', 'f'],
          'odor': ['p', 'a', 'l', 'n', 'f', 'c', 'y', 's', 'm'],
          'gill-attachment': ['f', 'a'],
          'gill-spacing': ['c', 'w'],
          'gill-size': ['n', 'b'],
          'gill-color': ['k', 'n', 'g', 'p', 'w', 'h', 'u', 'e', 'b', 'r', 'y', 'o'],
          'stalk-shape': ['e', 't'],
          'stalk-root': ['e', 'c', 'b', 'r', '?'],
          'stalk-surface-above-ring': ['s', 'f', 'k', 'y'],
          'stalk-surface-below-ring': ['s', 'f', 'y', 'k'],
          'stalk-color-above-ring': ['w', 'g', 'p', 'n', 'b', 'e', 'o', 'c', 'y'],
          'stalk-color-below-ring': ['w', 'p', 'g', 'b', 'n', 'e', 'y', 'o', 'c'],
          'veil-type': ['p'],
          'veil-color': ['w', 'n', 'o', 'y'],
          'ring-number': ['o', 't', 'n'],
          'ring-type': ['p', 'e', 'l', 'f', 'n'],
          'spore-print-color': ['k', 'n', 'u', 'h', 'w', 'r', 'o', 'y', 'b'],
          'population': ['s', 'n', 'a', 'v', 'y', 'c'],
          'habitat': ['u', 'g', 'm', 'd', 'p', 'w', 'l']}
         ```

         Do you notice something weird in the dictionary? Well, there are two columns that I had to deal with: `veil-type` and `stalk-root`.

         For `veil-type`, there is only one value instead of two, and for `stalk-root`, you will notice a `?` as one of the values in it—which isn't valid.

         According to my research, veil type isn't really a good indicator for edible mushrooms, so I decided to just drop the entire column.

         ```py
         df = df.drop(columns=["veil-type"])
         ```

         And for `stalk-root`, I just dropped rows with an invalid "?" character.

         ```
         df = df[df["stalk-root"] != "?"]
         ```

         Now, with invalid data removed, I was ready to encode all values in the dataset.

         ```py
         for column in df.columns:
             data_encoder = make_data_encoder(column)
             df[column] = df[column].map(data_encoder)
         ```

         Alright, let's see how does our dataset look.
         """)

st.image(img.open("images/ml-after-encode.png"))

st.write("""
         Nice! Now, they're all a number and pretty much clean.

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

         #### Models Used and Theorem
         The models that I will be using here are **Decision Tree** model and **Support Vector Machine** (SVM), which will both be wrapped with **Ensemble method**.

         Both methods that I'm using here are supervised models.

         ##### Decision Tree
         
         Source: https://www.geeksforgeeks.org/decision-tree/

         Decision Tree is the most explainable machine learning model amongst the other algorithm; you can even make a diagram for it. The model itself works with many kinds of data like non-linear relationships. Feature Scaling is also not required. However, this algorithm can lead to overfitting or bias that focuses too much on features with many catergories.
         
         It is represented as a hierarchical tree containing these elements:
         * Root Node: The big main question of the incoming problem and the entire point that represents the entire dataset.
         * Branches: The lines that are connecting nodes, showing the flow from one decision to another.
         * Internal Nodes: Points where decisions are made based on the input features.
         * Leaf Node: The terminal nodes at the end of the tree that represents the prediction or the answer.
         
         When the model receives an input, it starts with a big main question at the top of the tree with its branches linking to two other smaller questions as the model answer the incoming question—yes or no. This keeps on going until the last question with its branches linking to the answer.

         ##### Support Vector Machine (SVM)

         Source: https://medium.com/@RobuRishabh/support-vector-machines-svm-27cd45b74fbb

         Support Vector Machine (SVM) can perform very well when its hyperparameter is tuned right while not using as much memory. However, it may take a lot of modeling time, and it doesn't perform well with overlapping classes.

         The idea of SVM is to find the best margin (C) between a hyperplane—a decision boundary—and data points that are the closest to it called "support vectors."

         You can tune the margin how much are you going to allow the model to make some errors to accommodate with the unseen data and not being too familiar with the training one. This process is called "Regularization" or "Generalization."

         #### Modeling

         Now let's make ourselves machine learning models. Starting with Decision Tree...

         ```py
         from sklearn.tree import DecisionTreeClassifier
         
         d3_clf = DecisionTreeClassifier()
         d3_clf.fit(x_train, y_train)
         ```

         ...and followed by Support Vector Machine.

         ```py
         from sklearn.svm import SVC

         svm_clf = SVC()
         svm_clf.fit(x_train, y_train)
         ```

         But before we move on to wrapping them together, let's evaluate them first.

         ```py
         from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, confusion_matrix
         
         d3_pred = d3_clf.predict(x_test)
         svm_pred = svm_clf.predict(x_test)
         
         d3_accuracy = accuracy_score(y_test, d3_pred)
         d3_precision = precision_score(y_test, d3_pred)
         d3_recall = recall_score(y_test, d3_pred)
         d3_f1 = f1_score(y_test, d3_pred)
         d3_confusion_matrix = confusion_matrix(y_test, d3_pred)
         
         svm_accuracy = accuracy_score(y_test, svm_pred)
         svm_precision = precision_score(y_test, svm_pred)
         svm_recall = recall_score(y_test, svm_pred)
         svm_f1 = f1_score(y_test, svm_pred)
         svm_confusion_matrix = confusion_matrix(y_test, svm_pred)
         ```

         As you can see right here, I used testing data for both models to predict before finding their accuracy, precision, recall, F1 score, and their confusion matrix.

         And the result... was surprisingly shocking.         
         """)

st.image(img.open("images/ml-after-evaluation.png"))

st.write("""
         Both models got 100% on every metrics, leaving no false predictions.

         I'll be honest, at the first time when I got this much score, I was skeptical. I didn't believe that the model would perform this good. But consider the fact that the value types in the dataset are pretty simple, it kind of makes sense.

         Some times later, I learned about the `pd.factorize()` pandas method, and tried them on my notebook. However, the models seemed to perform worse for some reason, so I kept it in the same way.

         Let's get back to the point. I then started modeling the Ensemble method, wrapping both models that I mentioned into it.

         ```py
         from sklearn.ensemble import VotingClassifier
         
         estimators = [("decision_tree", d3_clf),
                       ("svm", svm_clf)]
         
         ensemble_clf = VotingClassifier(estimators)
         ensemble_clf.fit_transform(x_train, y_train)
         ```

         I then evaluated the model and the result is pretty much the same, so I won't mention it here.

         #### Try Demo

         And that's about it for how I model and evaluate my AI models for predicting mushrooms edibility.
         """)

st.page_link("pages/machine-learning-demo.py", label="Try it out here!")