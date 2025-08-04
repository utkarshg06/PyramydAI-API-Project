# Use Python base image
FROM python:3.9-slim

# Set working directory 
WORKDIR /Pyramyd

# Copy 
COPY . /Pyramyd

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python libraries
RUN pip install --no-cache-dir \
    nltk \
    fastapi \
    uvicorn \
    pandas \
    sentence-transformers \
    keybert


# Set environment variable for dataset path
ENV DATA_PATH="G2 software - CRM Category Product Overviews.csv"

# port
EXPOSE 8000

# Run the FastAPI app
CMD ["uvicorn", "Pyramyd:app", "--host", "0.0.0.0", "--port", "8000"]

