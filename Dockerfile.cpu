# base image
FROM python:3.12-slim-bullseye

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY dev dev
COPY serve serve
COPY run_server.sh .
# COPY .env .

# Make run_server.sh executable
RUN chmod +x run_server.sh

# Expose the port the app runs on
EXPOSE 8000

# Set environment variable for the port
ENV PORT=8000

# Command to run the FastAPI app
CMD ["./run_server.sh"]
