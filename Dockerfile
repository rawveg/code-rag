FROM python:3.9-slim

WORKDIR /app

# Install git and build essentials for sentence-transformers
RUN apt-get update && \
    apt-get install -y git build-essential && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the sentence transformer model
RUN python -c "from sentence_transformers import SentenceTransformer; \
    model = SentenceTransformer('all-MiniLM-L6-v2'); \
    print('Model downloaded successfully')"

COPY . .

CMD ["python", "app.py"]
