# Use Python slim image for smaller size
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements first (for layer caching)
COPY requirements-docker.txt .

# Upgrade pip and install dependencies with better timeout handling
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --timeout 300 -r requirements-docker.txt

# Copy the rest of the project
COPY . .

# Expose the API port
EXPOSE 8000

# Run the FastAPI app
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
