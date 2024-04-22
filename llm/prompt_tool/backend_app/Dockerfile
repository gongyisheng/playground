# Description: Dockerfile for the prompt tool backend app
FROM python:3.8.10

# Copy the current directory contents into the container
COPY . /opt/prompt_tool/backend_app

# Set the working directory
WORKDIR /opt/prompt_tool/backend_app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port the app runs on
EXPOSE 5600

# Run the application
CMD ["python3", "server.py", "--config", "/etc/prompt_tool/backend-config.yaml"]
