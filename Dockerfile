FROM python:3.12-slim

WORKDIR /app

COPY requirements-app.txt .
RUN pip install --no-cache-dir -r requirements-app.txt

COPY data/ data/
COPY src/ src/
COPY tests/ tests/
COPY app.py .

# Le modèle est réentraîné dans l'image pour correspondre à la version de scikit-learn installée
RUN python src/vin_qualite.py && python -m pytest -q

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
