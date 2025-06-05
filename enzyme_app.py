import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Load the trained model
model = pickle.load(open('enzyme_model.pkl', 'rb'))

# Load the dataset
df = pd.read_csv('synthetic_dataset.csv')

# Title of the app
st.title('🧪 Enzyme Activity Prediction and Analysis App')

st.markdown("""
Welcome to the Enzyme Prediction App!  
Fill in the details below to predict enzyme activity and visualize relationships 📈
""")

# User inputs for prediction
st.header("Input Parameters")

substrate_concentration = st.number_input('Enter Substrate Concentration (mol/L)', min_value=0.1, max_value=2.0, step=0.1)
inhibitor_concentration = st.number_input('Enter Inhibitor Concentration (mol/L)', min_value=0.0, max_value=0.5, step=0.01)
temperature = st.number_input('Enter Temperature (°C)', min_value=20, max_value=70, step=1)
pH = st.number_input('Enter pH level (4-9)', min_value=4.0, max_value=9.0, step=0.1)

# Create input dataframe
input_data = pd.DataFrame({
    'temperature': [temperature],
    'pH': [pH],
    'substrate_concentration': [substrate_concentration],
    'inhibitor_concentration': [inhibitor_concentration]
})

# Button to trigger prediction
if st.button('Predict Enzyme Activity'):
    prediction = model.predict(input_data)
    st.success(f"🎯 Predicted Enzyme Activity: {prediction[0]:.2f} units")

    # Show input and prediction in table
    st.subheader("Your Inputs and Prediction")
    result_df = input_data.copy()
    result_df['Predicted_Enzyme_Activity'] = prediction[0]
    st.dataframe(result_df)

    #st.balloons()

# Divider
st.markdown("---")

# Explore the dataset
st.header("📊 Explore the Synthetic Dataset")

if st.checkbox('Show Synthetic Dataset'):
    st.dataframe(df)

if st.checkbox('Show Correlation Heatmap'):
    st.subheader("Feature Correlation")
    plt.figure(figsize=(8,6))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    st.pyplot(plt)

if st.checkbox('Show pH vs Enzyme Activity Graph'):
    st.subheader("pH vs Enzyme Activity")
    plt.figure(figsize=(8,6))
    sns.scatterplot(x='pH', y='enzyme_activity', data=df, color='green')
    plt.title('Effect of pH on Enzyme Activity')
    st.pyplot(plt)

if st.checkbox('Show Temperature vs Enzyme Activity Graph'):
    st.subheader("Temperature vs Enzyme Activity")
    plt.figure(figsize=(8,6))
    sns.scatterplot(x='temperature', y='enzyme_activity', data=df, color='red')
    plt.title('Effect of Temperature on Enzyme Activity')
    st.pyplot(plt)

