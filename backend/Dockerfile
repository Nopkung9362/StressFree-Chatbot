# Use NVIDIA's official PyTorch image as the base
# This ensures all CUDA drivers and libraries are compatible
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Add the project root directory to the Python path
ENV PYTHONPATH "${PYTHONPATH}:/code"

# Set the working directory
WORKDIR /code

# Install dependencies
COPY ./requirements.txt /code/requirements.txt

# Update pip and install requirements
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r /code/requirements.txt

# Copy the application code
COPY ./app /code/app

# Command to run the application (The Correct Version)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]