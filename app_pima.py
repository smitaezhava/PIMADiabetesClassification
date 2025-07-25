import pandas as pd
import pickle
import torch
from fastapi import FastAPI
import sklearn
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Any

device = "cuda" if torch.cuda.is_available() else "cpu"

model_file = 'C:/Smita/AIML/PyTorch/PimaDiabetesClassification/ml_model_pimadiabetes.pkl'
loaded_model = pickle.load(open(model_file, 'rb'))

app = FastAPI()

@app.get('/')
def home():
    return 'PIMA Diabetes Classification!'
@app.post('/predict/')
def predict(data: List[Dict[str, Any]]):

  df_X = pd.DataFrame(data)

  df_X['skin'] = df_X['skin'].replace(0, df_X['skin'].median())
  df_X['mass'] = df_X['mass'].replace(0, df_X['mass'].median())
  df_X['Plas'] = df_X['Plas'].replace(0, df_X['Plas'].median())
  df_X['Pres'] = df_X['Pres'].replace(0, df_X['Pres'].median())
  df_X['test'] = df_X['test'].replace(0, df_X['test'].median())

  scaler=StandardScaler()
  X_std=scaler.fit_transform(df_X)

  prediction=loaded_model.predict(X_std)

  return str(prediction[0])
      
if __name__ == '__main__':
    app.run()