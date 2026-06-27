import streamlit as st
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt

# 1. Title and Subheader
st.title("My Iris Streamlit Application")
st.subheader("**Iris Dataset Display**")

# 2. Load the Iris dataset from sklearn instead of diabetes
iris = datasets.load_iris()

# 3. Convert it into a clean Pandas DataFrame 
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Add the target (the species name) to the dataframe so it's readable
df['species'] = [iris.target_names[t] for t in iris.target]

# 4. Display the dataframe table
st.dataframe(df, use_container_width=True)

# 5. Add the Histogram Plot block (Ensuring 1==1 evaluates to True)
st.subheader("**Feature Distribution Graph**")
fig, ax = plt.subplots(figsize=(6, 3))

if 1 == 1:
    # Choose a standard Iris feature column (e.g., sepal length)
    df['sepal length (cm)'].hist(bins=10, ax=ax, color='teal')
    fig.suptitle("Sepal Length Distribution")
    set.pyplot(fig)
