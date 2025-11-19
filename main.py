import streamlit as st
import numpy as np
from sklearn.ensemble import IsolationForest
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

st.title("Anomaly Detection in Machine Data")
st.write("Upload a .xlsx file. We'll drop 'Power Status', 'Machine Status', and 'Timestamp' columns, detect anomalies using Isolation Forest, and plot each numerical column with anomalies highlighted.")

# File uploader
uploaded_file = st.file_uploader("Choose a .xlsx file", type="xlsx")

if uploaded_file is not None:
    try:
        # Load the dataset into a pandas DataFrame
        df = pd.read_excel(uploaded_file)
        st.write("Original Data Preview:")
        st.dataframe(df.head())
        
        # Drop the string features (as in your code)
        columns_to_drop = ['Power Status', 'Machine Status', 'Timestamp']
        for col in columns_to_drop:
            if col in df.columns:
                df.drop(col, axis=1, inplace=True)
        
        st.write("Data after dropping columns:")
        st.dataframe(df.head())
        
        # Check for numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) == 0:
            st.error("No numerical columns found after dropping. Please check your file.")
        else:
            # Scale the features
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(df)
            
            # Create an Isolation Forest model
            clf = IsolationForest(contamination=0.15)  # Contamination is the proportion of outliers in the data
            
            # Fit the model and predict anomalies
            clf.fit(scaled_data)
            predictions = clf.predict(scaled_data)
            
            # Anomalies are labeled as -1, normal points are labeled as 1
            anomaly_indices = np.where(predictions == -1)[0]
            
            st.write(f"Detected {len(anomaly_indices)} anomalies out of {len(df)} rows.")
            
            # Plot for each numerical column
            for i in numerical_cols:
                fig, ax = plt.subplots()
                ax.plot(df.index, df[i], c='r')  # Line plot
                ax.scatter(anomaly_indices, df[i].loc[anomaly_indices], c='b', marker='*')  # Anomalies
                ax.set_title(f"Anomalies in {i}")
                st.pyplot(fig)
                
    except Exception as e:
        st.error(f"An error occurred: {str(e)}. Please ensure the file is a valid .xlsx with the expected structure.")
