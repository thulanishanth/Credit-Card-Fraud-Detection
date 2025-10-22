import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_preprocess_data(path):
    # Load the dataset
    data = pd.read_csv(path)
    
    # Feature-target split
    X = data.drop(columns=['Class'])
    y = data['Class']
    
    # Scale data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # ✅ Return 5 values including scaler
    return X_train, X_test, y_train, y_test, scaler
