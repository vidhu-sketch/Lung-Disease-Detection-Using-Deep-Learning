FROM tensorflow/tensorflow:2.11.0

WORKDIR /app

# Copy project files
COPY . /app

# Install Python deps
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

ENV PORT=5000
EXPOSE 5000

# NOTE: pass OPENAI_API_KEY as a runtime secret; do NOT bake it into the image.
# Example build / run:
#   docker build -t lung-cancer-app:latest .
#   docker run -e OPENAI_API_KEY="sk-..." -p 5000:5000 lung-cancer-app:latest

# Use gunicorn as the production WSGI server
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app", "--workers", "1", "--threads", "2", "--timeout", "120"]
