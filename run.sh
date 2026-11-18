#!/bin/zsh
# Lance l'app. Au premier lancement : crée l'environnement Python et télécharge les modèles.
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  python3 -m venv .venv && .venv/bin/pip install -q --upgrade pip && .venv/bin/pip install -q -r requirements.txt
fi
./scripts/get-models.sh >/dev/null || { echo "Téléchargement des modèles impossible (connexion ?)"; exit 1; }
exec .venv/bin/python main.py "$@"
