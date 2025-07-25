import requests
import pandas as pd
import random
import sklearn
import json
from sklearn.preprocessing import StandardScaler

# Define the URL of your Flask API endpoint
url = 'http://127.0.0.1:8000/predict'

# Define the path to the file you want to send
file_path = 'C:/Smita/AIML/PyTorch/PimaDiabetesClassification/pima-indians-diabetes.csv'  # Replace with the actual path to your file
df_data=pd.read_csv(file_path)

scaler=StandardScaler()
X = df_data.iloc[:,0:8]
X_std = scaler.fit_transform(X)

idx=random.randrange(1, len(df_data)) # Select a random record for prediction purpose
df_one_row=pd.DataFrame(X_std[idx:idx+1,:])
print(df_data.iloc[idx:idx+1,:])#To compare with predicted output

try:

    response = requests.post(
        url,
        json=df_one_row.to_dict(orient='records'))
    
    print('Predicted class : ' + response.json())
except Exception as e:
    print(f"An unexpected error occurred: {e}")