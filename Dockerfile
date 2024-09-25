# Use an official Python runtime as a parent image
FROM python:3.7-slim-buster

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN apt-get update -y && apt-get install -y gcc
RUN pip install --no-cache-dir -r requirements.txt

# Make port 8080 available to the world outside this container
EXPOSE 8080

# Define environment variable for Flask
ENV FLASK_ENV=production

# Run app.py when the container launches, and bind to the correct port
CMD ["python", "app.py"]
