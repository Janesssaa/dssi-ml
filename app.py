import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

st.write("""
# Simple Iris Flower Prediction App
This app predicts the **Iris flower** type based on your inputs!
""")

# 1. Sidebar for User Inputs
st.sidebar.header('User Input Parameters')

def user_input_features():
    sepal_length = st.sidebar.slider('Sepal length (cm)', 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider('Sepal width (cm)', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('Petal length (cm)', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('Petal width (cm)', 0.1, 2.5, 0.2)
    
    data = {
        'sepal length (cm)': sepal_length,
        'sepal width (cm)': sepal_width,
        'petal length (cm)': petal_length,
        'petal width (cm)': petal_width
    }
    features = pd.DataFrame(data, index=[0])
    return features

df_input = user_input_features()

# Display the user input parameters
st.subheader('User Input parameters')
st.write(df_input)

# 2. Model Training (For a live demo, training on the fly; or load from your /models folder)
iris = datasets.load_iris()
X = iris.data
Y = iris.target

clf = RandomForestClassifier()
clf.fit(X, Y)

# 3. Inference / Prediction
prediction = clf.predict(df_input)
prediction_proba = clf.predict_proba(df_input)

# 4. Displaying Results
st.subheader('Class Labels and their corresponding index number')
st.write(pd.DataFrame({'Species': iris.target_names}))

st.subheader('Prediction')
st.write(f"**{iris.target_names[prediction[0]]}**")

st.subheader('Prediction Probability')
st.write(pd.DataFrame(prediction_proba, columns=iris.target_names))