# Bank_ML

Bank_ML is a machine learning project designed to predict whether a customer will subscribe to a term deposit with the bank. Using a trained model and a preprocessing pipeline, the app analyzes customer data to provide real-time predictions. It is deployed as an interactive Gradio app on Hugging Face Spaces, allowing users to input data and receive instant predictions.


## Live Demo

Try the live deployment of this project on Hugging Face Spaces:

[![Hugging Face Spaces](https://img.shields.io/badge/🤗%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/luqmanabban/bank_prediction)

🔗 **Direct Link**: [https://huggingface.co/spaces/luqmanabban/bank_prediction](https://huggingface.co/spaces/luqmanabban/bank_prediction)

## Deployment

This project is deployed as a Gradio app on Hugging Face Spaces.  
It uses the following:
- Trained model: (best_hyperparameters.pkl)
- Preprocessing pipeline
- Interactive UI for predictions

## Deployment Details

This project is hosted on Hugging Face Spaces using Gradio. To run it locally:
```bash
pip install -r requirements.txt
Bank_ML.py

### **Commit and Push Changes**
Save your changes to `README.md`, commit, and push to GitHub:
```bash
git add README.md
git commit -m "Added Hugging Face Spaces deployment details"
git push origin main
