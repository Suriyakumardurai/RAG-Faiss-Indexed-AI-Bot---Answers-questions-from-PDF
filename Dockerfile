# Use lightweight Python base
FROM python:3.10-slim

# Set work directory
WORKDIR /app

# Prevents Python from buffering stdout/stderr
ENV PYTHONUNBUFFERED=1

# Install system dependencies (if needed by faiss / pdf libs)
RUN apt-get update && apt-get install -y \
    build-essential \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variable (Groq API Key will be injected at runtime)
ENV GROQ_API_KEY="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"

# Expose port (FastAPI default: 8000)
EXPOSE 8000

# Run app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
