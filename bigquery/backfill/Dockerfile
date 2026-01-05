FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY bigquery/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy backfill code
COPY bigquery/backfill/ /app/backfill/
COPY bigquery/shared/ /app/shared/
COPY bigquery/migrations/ /app/migrations/

# Set Python path
ENV PYTHONPATH=/app

# Run backfill script
CMD ["python", "-m", "backfill.main"]

