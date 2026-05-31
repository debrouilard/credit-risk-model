# Use an explicit, lightweight official Python runtime base
FROM python:3.10-slim

# Set system optimization variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Establish isolated internal application directory
WORKDIR /app

# Install dependencies first (leverages Docker layer caching)
COPY requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy project workspace code layers inside the container
COPY ./src /app/src

# Expose network interface port for internal router mapping
EXPOSE 8000

# Fire up uvicorn application server gateway upon runtime execution launch
CMD ["uvicorn", "src.api.main.app", "--host", "0.0.0.0", "--port", "8000"]