import pandas as pd
import pickle
import torch
import json
from fastapi import FastAPI
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

  df_data=pd.DataFrame(data)

  prediction=loaded_model.predict(df_data)

  return str(prediction[0])
      
if __name__ == '__main__':
    app.run()