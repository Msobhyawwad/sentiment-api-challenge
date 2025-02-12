from fastapi import FastAPI
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import os

from optimum.onnxruntime import ORTModelForSequenceClassification

import redis
import json

from pydantic import BaseModel

class PredictRequest(BaseModel):
    text: str


app = FastAPI()

model_name = "distilbert-base-uncased-finetuned-sst-2-english"

# Intentional bug: No token truncation for long texts
model = pipeline("text-classification", model=model_name)

tokenizer = AutoTokenizer.from_pretrained(model_name)

def chunk_text(text, max_length=512):
    tokens = tokenizer.encode(text, truncation=False, add_special_tokens=False)
    return [tokenizer.decode(tokens[i:i + max_length]) for i in range(0, len(tokens), max_length)]

# OR use the truncate and padding built-in options.
# model = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english", 
#                  tokenizer="distilbert-base-uncased-finetuned-sst-2-english", truncation=True, padding=True)

redis_client = redis.Redis(host="localhost", port=6379, db=0)

def get_cached_result(text: str):
    """
    Check if the result for the given text is cached in Redis.
    """
    print('Getting Cached results')
    cached_result = redis_client.get(text)
    if cached_result:
        return json.loads(cached_result)
    return None

def set_cached_result(text: str, result):
    """
    Cache the result for the given text in Redis.
    """
    print('Setting cache')
    redis_client.set(text, json.dumps(result))

@app.post("/predict")
async def predict(request: PredictRequest):
    text = request.text  # Extract text from the request body
    cached_result = get_cached_result(text)
    if cached_result:
        return cached_result
    chunks = chunk_text(text, max_length=512)

    results = [model(chunk) for chunk in chunks]
    set_cached_result(text, results)

    return results


## Quantized endpoint
## Fine-tuned approach

# fined_tuned_gpt2_name = "lvwerra/gpt2-imdb" 
# fined_tuned_gpt2 = AutoModelForSequenceClassification.from_pretrained(fined_tuned_gpt2_name)
# tokenizer_quantized = AutoTokenizer.from_pretrained(fined_tuned_gpt2)

# model_quantized = pipeline("text-classification", model=fined_tuned_gpt2, tokenizer=tokenizer_quantized)

# @app.post("/predict_quantized")
# async def predict(text: str):

#     results = model_quantized(text)
    
#     return results

## Use current BERT Quantized version

model_name_q = "Intel/distilbert-base-uncased-finetuned-sst-2-english-int8-static"
int8_model = ORTModelForSequenceClassification.from_pretrained(model_name_q)
model_q = pipeline("text-classification", model=int8_model, tokenizer=tokenizer)

@app.post("/predict_quantized")
async def predict_quantized(request: PredictRequest):
    text = request.text
    cached_result = get_cached_result(text)
    if cached_result:
        return cached_result
    chunks = chunk_text(text, max_length=512)

    results = [model_q(chunk) for chunk in chunks]
    set_cached_result(text, results)
    
    return results
