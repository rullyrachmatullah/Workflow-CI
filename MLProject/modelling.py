import pandas as pd
import mlflow
import mlflow.sklearn
import os
import dagshub

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

#INISIALISASI DAGSHUB & MLFLOW
dagshub.init(
    repo_owner="rullyrachmatullah",
    repo_name="Eksperimen_SML_RullyRachmatullah",
    mlflow=True
)

mlflow.set_experiment("insurance-exp")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, "med_clean.csv")

with mlflow.start_run() as run:
    mlflow.sklearn.autolog()

    df = pd.read_csv(data_path)

    X = df.drop("expenses", axis=1)
    y = df["expenses"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    #MANUAL METRIC (ADVANCE)
    mlflow.log_metric("mse_manual", mean_squared_error(y_test, y_pred))
    mlflow.log_metric("r2_manual", r2_score(y_test, y_pred))

    #MODEL
    mlflow.sklearn.log_model(model, artifact_path="model")

    #KUNCI CI/CD
    with open("run_id.txt", "w") as f:
        f.write(run.info.run_id)

    #TAG (nilai plus)
    mlflow.set_tag("pipeline", "ci")
    mlflow.set_tag("source", "github-actions")

print("✅ Training selesai | Logged ke DagsHub | CI-ready")
