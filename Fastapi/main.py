import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import pickle

app = FastAPI()

class ScoringItem(BaseModel):
    YearsAtCompany: float
    EmployeeSatisfaction: float

with open('rfmodel.pkl', 'rb') as f:
    model = pickle.load(f)

@app.post('/')
async def scoring_endpoint(item:ScoringItem):
    df = pd.DataFrame([item.dict().values()], columns = item.dict())
    yhat = model.predict(df)

    return {'prediction' : yhat}