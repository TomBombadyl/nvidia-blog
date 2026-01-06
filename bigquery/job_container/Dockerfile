FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code (build context is repo root)
COPY bigquery/job_container/sync_job.py .
COPY bigquery/mcp_server/large_content_handler.py ./large_content_handler.py

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run the sync job
CMD ["python", "sync_job.py"]
