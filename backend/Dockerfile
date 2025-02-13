FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

# Install necessary dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-setuptools \
    nvidia-docker2 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# Copy your application code
COPY . /usr/src/app
WORKDIR /usr/src/app

# Expose the port your app runs on
EXPOSE 7861

# Command to run your application
CMD ["python3", "app.py"]
