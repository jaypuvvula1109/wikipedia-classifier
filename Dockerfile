FROM rayproject/ray:2.9.0-py310-aarch64

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY models/ ./models/
COPY serving/serve.py ./serve_app.py

EXPOSE 8000

CMD ["python", "serve_app.py"]