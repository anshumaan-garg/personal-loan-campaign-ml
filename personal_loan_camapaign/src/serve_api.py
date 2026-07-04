"""
Production API using FastAPI + Ray Serve.
"""

import joblib
import numpy as np

from fastapi import FastAPI
from pydantic import BaseModel

from ray import serve

from config import MODEL_PATH

app = FastAPI(
    title="Personal Loan Prediction API",
    version="1.0.0",
)


class LoanRequest(BaseModel):

    Age: int
    Experience: int
    Income: float
    ZIP_Code: int
    Family: int
    CCAvg: float
    Education: int
    Mortgage: float
    Securities_Account: int
    CD_Account: int
    Online: int
    CreditCard: int


@serve.deployment
@serve.ingress(app)
class LoanModel:

    def __init__(self):

        self.model = joblib.load(MODEL_PATH)

    @app.get("/")
    async def root(self):

        return {
            "status": "healthy",
            "model": "Decision Tree",
        }

    @app.get("/health")
    async def health(self):

        return {"status": "UP"}

    @app.post("/predict")
    async def predict(
        self,
        request: LoanRequest,
    ):

        data = np.array(
            [
                [
                    request.Age,
                    request.Experience,
                    request.Income,
                    request.ZIP_Code,
                    request.Family,
                    request.CCAvg,
                    request.Education,
                    request.Mortgage,
                    request.Securities_Account,
                    request.CD_Account,
                    request.Online,
                    request.CreditCard,
                ]
            ]
        )

        prediction = int(
            self.model.predict(data)[0]
        )

        probability = float(
            self.model.predict_proba(data)[0][1]
        )

        return {
            "prediction": prediction,
            "probability": probability,
        }


deployment = LoanModel.bind()