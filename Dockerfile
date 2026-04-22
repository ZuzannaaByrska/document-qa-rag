# bazowy obraz Python
FROM python:3.11-slim

# ustaw folder roboczy w kontenerze
WORKDIR /app

# skopiuj requirements i zainstaluj biblioteki
# robimy to przed kopiowaniem kodu żeby Docker cachował tę warstwę
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# skopiuj cały kod do kontenera
COPY . .

# port na którym działa aplikacja
EXPOSE 8000

# komenda uruchomienia aplikacji
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]