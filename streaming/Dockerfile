# Use a specific Python version
FROM python:3.8

# Install system dependencies
RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx

WORKDIR /camera-ai
ENV PYTHONPATH /camera-ai

# Copy requirements and install dependencies
COPY ./requirements.txt /camera-ai/requirements.txt
RUN python -m pip install --upgrade pip==21.3.1 && \
    pip install --no-cache-dir --upgrade -r /camera-ai/requirements.txt


# Create a volume for site-packages
VOLUME ["/usr/local/lib/python3.8/site-packages"]

# Copy source code and other necessary files
COPY ./src /camera-ai/src
COPY ./logs /camera-ai/logs
COPY ./static /camera-ai/static
COPY ./wsdl /camera-ai/wsdl
COPY ./.env /camera-ai/.env


# Command to run the application
CMD ["python", "src/app.py"]
