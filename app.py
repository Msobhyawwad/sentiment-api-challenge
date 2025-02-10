from fastapi import FastAPI
from transformers import pipeline
import os

app = FastAPI()

# Intentional bug: No token truncation for long texts
model = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

@app.post("/predict")
async def predict(text: str):
    # Fails for texts >512 tokens
    return model(text)
