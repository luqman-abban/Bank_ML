import streamlit as st
import pandas as pd
from tensorflow.keras.models import load_model
import joblib

# Load model and hyperparameters
model = load_model('best_model.h5')
encoder = joblib.load('encoder.pkl')  # Assume you have a pre-trained encoder saved in joblib

# Preprocessing function
def preprocess_data(data):
    """
    Encodes the original text data into the format required by the model.
    Assumes `encoder` is a pre-fitted object with the necessary transformations.
    """
    # Example: Assume `encoder` transforms categorical features
    encoded_data = encoder.transform(data)
    return encoded_data

# Prediction function
def make_predictions(encoded_data):
    predictions = model.predict(encoded_data)
    binary_predictions = (predictions >= 0.5).astype(int)
    return ["yes" if pred == 1 else "no" for pred in binary_predictions]

# Streamlit app
st.title("Customer Prediction App")

# File upload
uploaded_file = st.file_uploader("Upload your CSV file with original data", type="csv")

if uploaded_file is not None:
    # Read file
    original_data = pd.read_csv(uploaded_file)
    st.write("Original Data:")
    st.write(original_data)
    
    # Preprocess and encode
    try:
        encoded_data = preprocess_data(original_data)
        st.write("Encoded Data:")
        st.write(pd.DataFrame(encoded_data))
        
        # Run predictions
        predictions = make_predictions(encoded_data)
        st.write("Predictions:")
        st.write(predictions)
        
        # Save predictions to CSV
        result_df = original_data.copy()
        result_df["Prediction"] = predictions
        result_csv = result_df.to_csv(index=False)
        
        # Provide download link
        st.download_button(
            label="Download Predictions as CSV",
            data=result_csv,
            file_name="predictions.csv",
            mime="text/csv",
        )
    except Exception as e:
        st.error(f"Error processing the file: {e}")
