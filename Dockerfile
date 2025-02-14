FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir pygments==2.17.2

# Copy the source code
COPY src/ /app/src/
COPY app.py .
COPY templates/ /app/templates/

# Update Python path to include src directory
ENV PYTHONPATH=/app

CMD ["python", "app.py"]
