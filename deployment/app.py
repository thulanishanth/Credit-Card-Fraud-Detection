import sys
import os
from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# ✅ Fix import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now this works fine
from src.data_preprocessing import load_preprocess_data
from src.model_evaluation import evaluate_model

def train_model():
    """
    Train a RandomForestClassifier for fraud detection and save model artifacts.
    """
    print("🚀 Starting model training...")

    # ✅ Fix dataset path
    csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'creditcard.csv'))
    X_train, X_test, y_train, y_test, scaler = load_preprocess_data(csv_path)

    # Define and train model
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight='balanced'  # handle class imbalance
    )

    model.fit(X_train, y_train)
    print("✅ Model training completed successfully!")

    # Evaluate performance
    evaluate_model(model, X_test, y_test)

    # Save model and scaler
    model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model'))
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(model_dir, "fraud_detection_model.pkl"))
    joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))
    print(f"💾 Model and scaler saved in '{model_dir}' directory!")


if __name__ == "__main__":
    train_model()
