import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class item(BaseModel):
    YearsAtCompany: float
    EmployeeSatisfaction: float


@app.get('/')
async def scoring_endpoint():
    return {'Hello' : 'Test'}