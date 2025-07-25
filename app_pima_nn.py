import pandas as pd
import pickle
import torch
from fastapi import FastAPI
import sklearn
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Any
from myModel import myModel

device = "cuda" if torch.cuda.is_available() else "cpu"

model_file = 'C:/Smita/AIML/PyTorch/PimaDiabetesClassification/nn_model_pimadiabetes.pt'

nn_model=myModel()
nn_model.load_state_dict(torch.load(model_file, weights_only=True))
nn_model.eval()


app = FastAPI()

@app.get('/')
def home():
    return 'PIMA Diabetes Classification!'
@app.post('/predict/')
def predict(data: List[Dict[str, Any]]):

  df_X = pd.DataFrame(data)

  prediction=nn_model(torch.tensor(df_X.to_numpy(), dtype = torch.float32).to(device))

  return str(round(prediction.item()))
      
if __name__ == '__main__':
    app.run()