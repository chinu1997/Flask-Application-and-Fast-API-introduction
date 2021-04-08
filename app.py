from flask import Flask, request,jsonify
import numpy as np
import pickle as p
import pandas as pd
import json
from fastapi import FastAPI

app=FastAPI()

modelfile = 'models/regression_final.pickle'

model = p.load(open(modelfile, 'rb'))


@app.get('/')
async def main():
    return ('Predict Boston House  API')


@app.post("/api/")
async def Make_Prediction():
    try:
        j_data = request.get_json()
        try:
            prediction = np.array2string(model.predict(j_data))
            return {"message":jsonify(prediction)}
        except:
            return "check your Model Input Again"
    except:
        return "Check your json data it should be like this format [[12,25,45,14,78,12,45]]"

