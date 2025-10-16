# Utilise une image officielle Python 3.12 slim (base Debian bullseye)
FROM python:3.12-slim

# Installer les dépendances système nécessaires pour compiler (gcc, build-essential, python-dev, etc)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    python3-dev \
    cython3 \
    && rm -rf /var/lib/apt/lists/*

# Créer un dossier pour l'app
WORKDIR /app

# Copier les fichiers requirements et l'application dans l'image
COPY requirements.txt .

# Mettre à jour pip, setuptools et wheel (meilleure gestion des builds)
RUN pip install --upgrade pip setuptools wheel

# Installer d'abord Cython explicitement pour le build de portmin
RUN pip install Cython

# Installer les autres dépendances listées dans requirements.txt
RUN pip install -r requirements.txt

# Copier tout le code source dans /app
COPY . .

# Exposer le port utilisé par Flask (par défaut 8000 ou 5000)
EXPOSE 8000

# Définir la commande de démarrage, par exemple avec gunicorn (adapter selon ton app)
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--timeout", "120", "app:app"]

