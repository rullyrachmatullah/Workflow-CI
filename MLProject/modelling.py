import pandas as pd
import mlflow
import mlflow.sklearn
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

mlflow.autolog()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "med_clean.csv")

df = pd.read_csv(DATA_PATH)

X = df.drop("expenses", axis=1)
y = df["expenses"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# 🔑 INI KUNCI
mlflow.sklearn.log_model(
    model,
    name="model",
    registered_model_name="insurance-model"
)

print("Training selesai via MLflow Project 🚀")
