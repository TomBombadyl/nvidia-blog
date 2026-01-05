FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY bigquery/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy incremental code
COPY bigquery/incremental/ /app/incremental/
COPY bigquery/shared/ /app/shared/
COPY bigquery/migrations/ /app/migrations/

# Set Python path
ENV PYTHONPATH=/app

# Run incremental sync script
CMD ["python", "-m", "incremental.main"]

