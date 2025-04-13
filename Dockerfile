# Stage 1: Base Python environment and install dependencies
FROM python:3.12-slim as builder

# Set working directory
WORKDIR /app

# Install build dependencies for PyAudio
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libportaudio2 \
    libportaudiocpp0 \
    portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

# Prevent Python from writing pyc files
ENV PYTHONDONTWRITEBYTECODE 1
# Ensure Python output is sent straight to terminal
ENV PYTHONUNBUFFERED 1

# Copy requirements first to leverage Docker cache
COPY requirements.txt ./

# Install Python dependencies (Ensure gunicorn is in requirements.txt)
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data during the build
RUN python -m nltk.downloader punkt


# Stage 2: Final application image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install runtime dependencies (libportaudio2)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libportaudio2 \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user and group
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser

# Copy installed dependencies from the builder stage
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
# Copy executables (like gunicorn) installed by pip
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy NLTK data from the builder stage
COPY --from=builder /root/nltk_data /home/appuser/nltk_data
# Set NLTK_DATA environment variable for the app user
ENV NLTK_DATA=/home/appuser/nltk_data

# Copy application code
COPY --chown=appuser:appgroup . .

# *** FIX: Explicitly add /usr/local/bin to the PATH for the appuser ***
ENV PATH="/usr/local/bin:${PATH}"

# Switch to the non-root user
USER appuser

# Expose the port the app will run on
EXPOSE 8080

# Define the command to run the application using gunicorn and eventlet
CMD ["python", "app.py"]