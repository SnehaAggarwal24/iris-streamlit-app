import streamlit as st
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

# Load model
with open('iris_model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("🌸 Iris Flower Predictor")
st.markdown("Enter flower measurements and get predictions!")

st.sidebar.header("🌼 Input Features")
sepal_length = st.sidebar.slider("Sepal Length (cm)", 4.0, 8.0, 5.4)
sepal_width = st.sidebar.slider("Sepal Width (cm)", 2.0, 4.5, 3.4)
petal_length = st.sidebar.slider("Petal Length (cm)", 1.0, 7.0, 1.3)
petal_width = st.sidebar.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

if st.button("Predict"):
    prediction = model.predict(input_data)
    flower_types = ['Setosa', 'Versicolor', 'Virginica']
    st.success(f"Predicted Flower Type: **{flower_types[prediction[0]]}**")

if st.checkbox("Show Dataset and Plot"):
    iris_df = sns.load_dataset("iris")
    st.dataframe(iris_df.head())

    fig, ax = plt.subplots()
    sns.scatterplot(data=iris_df, x="sepal_width", y="petal_width", hue="species", ax=ax)
    st.pyplot(fig)
