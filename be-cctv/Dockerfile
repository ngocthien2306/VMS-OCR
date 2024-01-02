# Use a specific Python version
FROM python:3.8

WORKDIR /be-cctv
ENV PYTHONPATH /be-cctv

# Copy requirements and install dependencies
COPY ./requirements.txt /be-cctv/requirements.txt
RUN python -m pip install --upgrade pip==21.3.1 && \
    pip install --no-cache-dir --upgrade -r /be-cctv/requirements.txt

# Create a volume for site-packages
VOLUME ["/usr/local/lib/python3.8/site-packages"]

# Copy source code and other necessary files
COPY ./src /be-cctv/src
COPY ./public /be-cctv/public
COPY ./static /be-cctv/static
COPY ./.env /be-cctv/.env
# Command to run the application
CMD ["python", "src/app.py"]
