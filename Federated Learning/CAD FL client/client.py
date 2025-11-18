import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import requests

client_id = "client1"
data_file = "CAD1.csv" 
server_url = "http://192.168.235.107:5000/upload/" + client_id  

df = pd.read_csv(data_file)
X = df.drop('CAD', axis=1).values
y = df['CAD'].values

model = LogisticRegression(max_iter=1000)
model.fit(X, y)

np.save("weights_coef.npy", model.coef_)
np.save("weights_intercept.npy", model.intercept_)

files = {
    'coef': open('weights_coef.npy', 'rb'),
    'intercept': open('weights_intercept.npy', 'rb')
}

try:
    response = requests.post(server_url, files=files)
    print(response.text)
except Exception as e:
    print(f"Upload failed: {e}")
