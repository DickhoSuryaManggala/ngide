# Use Python 3.13 as base
FROM python:3.13-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-tk \
    libtcl8.6 \
    libtk8.6 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir pandas numpy

# Copy the rest of the application
COPY . .

# Command to run (GUI will need X11 forwarding if run in Docker, 
# but we can run backtests or the bot in headless mode if needed)
CMD ["python", "trading_app.py"]
