# Use Python 3.8 slim image
FROM python:3.8-bullseye

# Set the working directory to /app
WORKDIR /app

# Copy the local src directory to /app in the container
COPY ./src /app
RUN pip install -r requirements.txt

EXPOSE 5000

# The default command to run, you can modify this according to your needs
CMD ["python", "api.py"]
