# Sentiment Analysis API Challenge

A FastAPI service for sentiment analysis using a pre-trained model. This service is under active development and requires optimization for production use.

## Overview
This API provides a `/predict` endpoint to analyze sentiment (positive/negative) of input text. The current implementation has **known stability and performance issues** that need resolution.

## Prerequisites
- Python 3.13+
- Docker
- Git

## Installation
1. Clone the repo:
   ```bash
   git clone https://github.com/Msobhyawwad/sentiment-api-challenge.git
   cd sentiment-api-challenge
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Build Docker image (may require troubleshooting):
   ```bash
   docker build -t sentiment-api .
   ```

## Usage
1. Run locally:
   ```bash
   uvicorn app:app --reload
   ```
2. Run via Docker:
   ```bash
   docker run -p 80:80 sentiment-api
   ```
3. Test the endpoint:
   ```bash
   curl -X POST "http://localhost:80/predict" -H "Content-Type: application/json" -d '{"text": "This movie was fantastic!"}'
   ```

## Known Issues
- The `/predict` endpoint fails for long input texts (>500 words).
- Docker build may fail due to environment inconsistencies.

## Task Requirements
1. Fix the Docker build process.
2. Resolve the token limitation error for long texts.
3. (Optional) Improve inference latency.
4. Add tests to validate your fixes.
5. (Bonus) Add CI/CD for automated testing.

## Evaluation Criteria
- Code quality and readability
- Effectiveness of debugging
- Scalability considerations
- Documentation clarity

## Submission
1. Fork this repo.
2. Implement your solution.
3. Include a `SOLUTION.md` explaining:
   - How you diagnosed/fixed issues
   - Trade-offs made during optimization
4. (Optional) Add a 2-minute Loom video demo.
5. Share the repo link within **48 hours**.

## Bonus Points
- Add caching for repeated requests
- Implement model quantization
- Add monitoring/metrics


## Solving Large Token Issues
The given model's [documentation](https://huggingface.co/distilbert/distilbert-base-uncased#:~:text=The%20only%20constrain%20is%20that%20the%20result%20with%20the%20two%20%22sentences%22%20has%20a%20combined%20length%20of%20less%20than%20512%20tokens.) sets the limit of tokens to 512. To solve this issue we only have two options:

> Truncate incoming input

> Wrap in chunks 

I have commented the part for the first solution where I used the built-in parameters "truncate" and "padding". However, this is not the best option as it will some information. the other path is implemented in chunks function.


## Quantization and Distillation
The given model is the distilled version of Bert. To quantize it, I used the quantized version of it to improve latency.

I created a new endpoint called `/predict_quantized` for easier assessment of latency.

## Tests
To run the tests, use the following command:
> pytest test_app.py -v

