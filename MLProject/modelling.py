import pandas as pd
import mlflow
import mlflow.sklearn
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# ==============================
# AUTOLOG (WAJIB, TANPA START_RUN)
# ==============================
mlflow.autolog()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "med_clean.csv")

# ==============================
# Load data
# ==============================
df = pd.read_csv(DATA_PATH)

X = df.drop("expenses", axis=1)
y = df["expenses"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==============================
# Model
# ==============================
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# ==============================
# Log model KE RUN (BUKAN REGISTRY)
# ==============================
mlflow.sklearn.log_model(
    sk_model=model,
    artifact_path="model"
)

print("Training selesai")
