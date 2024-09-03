FROM python:3.10.12-slim

# Update and install curl
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY . .

# Expose port 5000 for the Flask application
EXPOSE 5000


RUN useradd -m appuser
USER appuser

ENV PORT=5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]