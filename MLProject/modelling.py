import pandas as pd
import mlflow
import mlflow.sklearn
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "med_clean.csv")

# MLflow lokal (AMAN untuk CI & Docker)
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("insurance-exp")

with mlflow.start_run() as run:
    # autolog parameter, metric, model
    mlflow.sklearn.autolog()

    # load data
    df = pd.read_csv(DATA_PATH)

    X = df.drop("expenses", axis=1)
    y = df["expenses"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # model
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    # prediction
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    mlflow.log_metric("mse_manual", mse)
    mlflow.log_metric("r2_manual", r2)

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model"
    )

    with open(os.path.join(BASE_DIR, "run_id.txt"), "w") as f:
        f.write(run.info.run_id)

    mlflow.set_tag("pipeline", "ci")
    mlflow.set_tag("source", "github-actions")

print("Training selesai | MLflow lokal | CI & Docker ready")
