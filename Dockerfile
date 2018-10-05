# Use an official Python runtime as a parent image
FROM python:3.6.5-slim

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN apt-get update
#RUN apt-get -y upgrade
RUN apt-get install libgtk2.0-dev -y
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV FLASK_APP app.py
#ENTRYPOINT ["python", "-m", "flask", "run", "--host=0.0.0.0"]

# Run app.py when the container launches
CMD ["python", "src/server/app.py"]
