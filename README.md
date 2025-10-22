# 💳 Credit Card Fraud Detection

A mini machine learning project to detect fraudulent credit card transactions.

## 🚀 Features
- End-to-end ML workflow (preprocessing → training → evaluation → deployment)
- Flask web app for prediction
- RandomForest model with class balancing
- Modular folder structure

## 🏗 How to Run
1. Clone this repository  
2. Place `creditcard.csv` inside `/data`  
3. Install dependencies  
   ```bash
   pip install -r requirements.txt
4. Train the model

python src/model_training.py

5. Run Flask app

python deployment/app.py

6. Open browser → http://127.0.0.1:5000


---

## ⚙️ Run Sequence

```bash
cd Credit-Card-Fraud-Detection
pip install -r requirements.txt
python src/model_training.py
python deployment/app.py


Dataset Download Instructions

🧾 1️⃣ Dataset: data/creditcard.csv

You can download the real dataset (used by thousands of ML learners) from Kaggle:

🔗 Credit Card Fraud Detection Dataset (Kaggle)

After download:

File name: creditcard.csv

Place it in your data/ folder.
