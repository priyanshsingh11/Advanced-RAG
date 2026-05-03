# Use a full Python image to ensure all scientific dependencies build correctly
FROM python:3.11

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies (needed for some PDF processing and C++ extensions)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    gfortran \
    libopenblas-dev \
    pkg-config \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY ./app ./app
COPY .env .env
# Create a data folder for document storage
RUN mkdir -p /app/data /app/storage

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
# We use 0.0.0.0 to allow external connections to the container
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
