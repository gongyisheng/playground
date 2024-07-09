# Description: Dockerfile for the prompt tool backend app
FROM python:3.8.10

# Copy the current directory contents into the container
COPY . /opt/lvs_case

# Set the working directory
WORKDIR /opt/lvs_case

# Install dependencies
RUN pip install --no-cache-dir redis

# Expose the port the app runs on
EXPOSE 5600

# Run the application
CMD ["python3", "polling.py"]