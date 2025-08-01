# Dockerfile pour Basket Stats
FROM python:3.11-slim

# Installer les dépendances système
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers de configuration
COPY pyproject.toml poetry.lock ./

# Installer Poetry et les dépendances
RUN pip install poetry && \
    poetry config virtualenvs.create false && \
    poetry install --no-dev

# Copier le code source
COPY . .

# Exposer le port de l'API
EXPOSE 8000

# Commande par défaut
CMD ["poetry", "run", "uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
