import gradio as gr
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

# Load the saved model and preprocessors
model = load_model('best_model.h5')
best_hyperparameters = joblib.load('best_hyperparameters.pkl')
scaler = StandardScaler()
encoder = LabelEncoder()

# Load the original data to get feature names and order
original_data = pd.read_csv('bank-full.csv', delimiter=';', quotechar='"')
categorical_cols = original_data.select_dtypes(include=['object']).columns

def predict_subscription(age, job, marital, education, default, balance, housing, loan, contact, day, month, duration, campaign, pdays, previous, poutcome):
    input_data = {
        'age': age, 'job': job, 'marital': marital, 'education': education, 'default': default,
        'balance': balance, 'housing': housing, 'loan': loan, 'contact': contact,
        'day': day, 'month': month, 'duration': duration, 'campaign': campaign,
        'pdays': pdays, 'previous': previous, 'poutcome': poutcome
    }
    user_df = pd.DataFrame([input_data])

    categorical_cols_to_encode = [col for col in categorical_cols if col in user_df.columns]
    for col in categorical_cols_to_encode:
        user_df[col] = encoder.fit_transform(user_df[col])

    numerical_cols = user_df.select_dtypes(include=['int64', 'float64']).columns
    user_df[numerical_cols] = scaler.fit_transform(user_df[numerical_cols])

    prediction = model.predict(user_df)
    binary_prediction = (prediction >= 0.5).astype(int)
    label_prediction = "yes" if binary_prediction[0][0] == 1 else "no"
    return label_prediction

# Define input components for Gradio
inputs = []
for col in original_data.columns[:-1]:
    if col in categorical_cols:
        unique_values = list(original_data[col].unique())
        # Change here: Use gr.<Component> directly
        inputs.append(gr.Dropdown(choices=unique_values, label=col)) 
    else:
        # Change here: Use gr.<Component> directly
        inputs.append(gr.Number(label=col))  

# Create the Gradio interface
iface = gr.Interface(
    fn=predict_subscription,
    inputs=inputs,
    outputs="text",
    title="Bank Marketing Outcome Prediction",
    description="Predict whether a customer will subscribe to a term deposit."
)

iface.launch()
