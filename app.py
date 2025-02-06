import uvicorn
from fastapi import FastAPI
import numpy as np
import pandas as pd
import pickle
from BankNotes import BankNote

app = FastAPI()

pickle_in = open("classifier.pkl", "rb")
classifier = pickle.load(pickle_in)


@app.get("/")
def index():
    return {"message": "Hello, Stranger"}


@app.get("/{name}")
def get_name(name: str):
    return {"message": f"Hello, {name}"}


@app.post("/predict")
def predict_bankNote(data: BankNote):
    data = data.dict()
    print(data)
    variance = data["variance"]
    skewness = data["skewness"]
    curtosis = data["curtosis"]
    entropy = data["entropy"]
    prediction = classifier.predict([[variance, skewness, curtosis, entropy]])
    if prediction >= 0.5:
        prediction = "Fake Note"
    else:
        prediction = "Its a Bank Note"

    return {"Prediction Score": f"{prediction}"}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
