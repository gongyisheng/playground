# Description: Dockerfile for the prompt tool backend app
FROM python:3.8.10

# Set the working directory
WORKDIR /

# Copy the current directory contents into the container
COPY . /

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

# Expose the port the app runs on
EXPOSE 5600

# Run the application
CMD ["sudo", "python", "server.py", "--config", "/etc/prompt_tool/backend-config.prod.yaml"]
