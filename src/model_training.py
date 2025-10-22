from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib
from src.data_preprocessing import load_preprocess_data
from src.model_evaluation import evaluate_model

def train_model():
    result = load_preprocess_data('data/creditcard.csv')
    X_train, X_test, y_train, y_test, *rest = result
    scaler = rest[0] if rest else None
    
    # Choose model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate the Model
    evaluate_model(model, X_test, y_test)
    
    # Save model and scaler
    joblib.dump(model, 'model/fraud_detection_model.pkl')
    if scaler is not None:
        joblib.dump(scaler, 'model/scaler.pkl')
    
if __name__ == "__main__":
    train_model()
