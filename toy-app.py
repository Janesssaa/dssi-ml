import streamlit as st
import pandas as pd
from sklearn import datasets

# Set application header
st.subheader('**Iris Dataset Display**')

# Load Iris dataset instead of diabetes
iris = datasets.load_iris()

# Create DataFrame with Iris features
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Optional: Add the target species names to make the table meaningful
df['species'] = [iris.target_names[i] for i in iris.target]

# Display dataframe as an interactive table
st.dataframe(df, use_container_width=True)