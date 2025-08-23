# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose the port that the app will run on
EXPOSE 5000

# Run the API server using gunicorn, a production-grade WSGI server
# The command assumes your Flask app is in a file named 'api.py'
# The `api:app` part means "look inside the `api.py` file for a Flask application object named `app`"
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "api:app"]