from flask import Flask, request
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
CLIENTS = ["client1", "client2"] 

for client in CLIENTS:
    os.makedirs(os.path.join(UPLOAD_FOLDER, client), exist_ok=True)


@app.route('/upload/<client_id>', methods=['POST'])
def upload_weights(client_id):
    if client_id not in CLIENTS:
        return "Invalid client ID", 400

    coef_file = request.files.get('coef')
    int_file = request.files.get('intercept')

    if not coef_file or not int_file:
        return "Missing files", 400

    coef_path = os.path.join(UPLOAD_FOLDER, client_id, "weights_coef.npy")
    int_path = os.path.join(UPLOAD_FOLDER, client_id, "weights_intercept.npy")

    coef_file.save(coef_path)
    int_file.save(int_path)

    return f"Weights successfully sent from {client_id}."


@app.route('/aggregate', methods=['GET'])
def aggregate_models():
    coefs, intercepts = [], []
    used_clients = []

    for client in CLIENTS:
        coef_path = os.path.join(UPLOAD_FOLDER, client, "weights_coef.npy")
        int_path = os.path.join(UPLOAD_FOLDER, client, "weights_intercept.npy")

        if os.path.exists(coef_path) and os.path.exists(int_path):
            coefs.append(np.load(coef_path))
            intercepts.append(np.load(int_path))
            used_clients.append(client)

    if not coefs:
        return "No client weights available for aggregation.", 400

    # Federated averaging
    avg_coef = np.mean(coefs, axis=0)
    avg_inter = np.mean(intercepts, axis=0)

    df = pd.read_csv("CAD3.csv")
    X = df.drop('CAD', axis=1).values
    y = df['CAD'].values

    model = LogisticRegression()
    model.coef_ = avg_coef
    model.intercept_ = avg_inter
    model.classes_ = np.array([0, 1])

    preds = model.predict(X)
    acc = accuracy_score(y, preds)

    output = f"""
Global Model Evaluation
-------------------------------
Clients Aggregated: {len(used_clients)}
Clients Used: {', '.join(used_clients)}

Final Global Weights
-------------------------------
Average Coefficients:
{avg_coef}

Average Intercept:
{avg_inter}

Evaluation on Full Dataset
-------------------------------
Accuracy: {acc:.4f}
"""

    return f"<pre>{output.strip()}</pre>"


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
