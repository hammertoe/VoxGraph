# Stage 1: Base Python environment and install dependencies
FROM python:3.12-slim as builder

# Set working directory
WORKDIR /app

# *** Install build dependencies ***
# Install build-essential (for gcc, make, etc.) and libportaudio2 + dev headers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libportaudio2 \
    libportaudiocpp0 \
    portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

# Prevent Python from writing pyc files
ENV PYTHONDONTWRITEBYTECODE 1
# Ensure Python output is sent straight to terminal (useful for container logs)
ENV PYTHONUNBUFFERED 1

# Copy requirements first to leverage Docker cache
COPY requirements.txt ./

# Install Python dependencies (PyAudio should build now)
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data during the build
RUN python -m nltk.downloader punkt


# Stage 2: Final application image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# *** Install runtime dependencies (libportaudio2) ***
# We only need the runtime library, not the dev headers or build tools, in the final image
RUN apt-get update && apt-get install -y --no-install-recommends \
    libportaudio2 \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user and group
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser

# Copy installed dependencies from the builder stage
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy NLTK data from the builder stage
# Find the nltk_data directory (location can vary slightly)
# Common locations: /usr/local/share/nltk_data, /root/nltk_data, /usr/share/nltk_data
# Adjust the source path if needed after inspecting the builder stage
COPY --from=builder /root/nltk_data /home/appuser/nltk_data
# Set NLTK_DATA environment variable for the app user
ENV NLTK_DATA=/home/appuser/nltk_data

# Copy application code
COPY --chown=appuser:appgroup . .

# Switch to the non-root user
USER appuser

# Expose the port the app will run on (Code Engine uses PORT env var, defaulting to 8080 is common)
EXPOSE 8080

# Define the command to run the application using gunicorn and eventlet
# It binds to 0.0.0.0 and port 8080 (Code Engine maps external traffic to this)
# Use -w 1 for eventlet worker with SocketIO
CMD ["gunicorn", "--worker-class", "eventlet", "-w", "1", "--bind", "0.0.0.0:8080", "app:socketio"]