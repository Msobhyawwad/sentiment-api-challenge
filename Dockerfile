FROM python:3.9

# Missing dependency: transformers not pinned, causing conflicts
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]
