FROM python:3.10.7-slim

# Prevent Python from writing .pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN apt-get update && apt-get install -y gcc \
    && pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get remove -y gcc && apt-get autoremove -y

# Copy app files
COPY . .

# Expose port
EXPOSE 5000

# Start the app using gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
