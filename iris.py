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
    # Dynamically grab the very first feature column name automatically
    first_column_name = iris.feature_names[0] 
    
    # Plot using that dynamic column key
    df[first_column_name].hist(bins=10, ax=ax, color='teal')
    fig.suptitle(f"{first_column_name.title()} Distribution")
    st.pyplot(fig)

    from sklearn.ensemble import RandomForestClassifier

st.write("---")
st.subheader("**Predict Iris Species**")

# 1. Train a quick model on the data
X = iris.data
y = iris.target
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# 2. Add interactive sliders in a sidebar or main layout for user input
st.write("Adjust the features below to make a live prediction:")
s_length = st.slider("Sepal Length (cm)", float(df.iloc[:,0].min()), float(df.iloc[:,0].max()), float(df.iloc[:,0].mean()))
s_width  = st.slider("Sepal Width (cm)",  float(df.iloc[:,1].min()), float(df.iloc[:,1].max()), float(df.iloc[:,1].mean()))
p_length = st.slider("Petal Length (cm)", float(df.iloc[:,2].min()), float(df.iloc[:,2].max()), float(df.iloc[:,2].mean()))
p_width  = st.slider("Petal Width (cm)",  float(df.iloc[:,3].min()), float(df.iloc[:,3].max()), float(df.iloc[:,3].mean()))

# 3. Create the input array for the model prediction
user_features = [[s_length, s_width, p_length, p_width]]

# 4. Make the prediction when the user is ready
if st.button("Predict Species"):
    prediction = model.predict(user_features)[0]
    predicted_species = iris.target_names[prediction]
    
    # Display the result beautifully
    st.success(f"The model predicts this flower is an **Iris {predicted_species.title()}**!")
