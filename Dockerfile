# Use official lightweight Python image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

# Set work directory
WORKDIR /workspace

# Install system dependencies (curl for healthchecks)
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

# Install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY app/ app/

# Expose port (Cloud Run sets this dynamically)
EXPOSE 8080

# Command to run application using shell execution to parse env variable $PORT
CMD exec uvicorn app.main:app --host 0.0.0.0 --port $PORT
