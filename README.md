# Sentiment Analysis API Challenge

A FastAPI service for sentiment analysis using a pre-trained model. This service is under active development and requires optimization for production use.

## Overview
This API provides a `/predict` endpoint to analyze sentiment (positive/negative) of input text. The current implementation has **known stability and performance issues** that need resolution.

## Prerequisites
- Python 3.9+
- Docker
- Git

## Installation
1. Clone the repo:
   ```bash
   git clone https://github.com/[your_username]/sentiment-api-challenge.git
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

