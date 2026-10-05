# Classification de la qualité du vin

Projet 2 – Machine Learning I – ING2-BDML – EFREI 2026/2027

Prédire si un vin est **bon (note ≥ 7)** à partir de 11 mesures physico-chimiques et de son type (rouge ou blanc).

## Résultats

| | |
|---|---|
| Données | 6 497 vins (1 599 rouges, 4 898 blancs) → 5 258 après nettoyage, 19,1 % de bons vins |
| Modèles comparés | Régression logistique, k-NN, arbre de décision, Naive Bayes, SVM linéaire, SVM RBF |
| Optimisation | Grid Search et Random Search, validation croisée stratifiée à 5 plis, critère F1 |
| **Modèle retenu** | **SVM RBF** (`C=100`, `gamma=0.01`, `class_weight='balanced'`) |
| Test (1 315 vins) | F1 = 0,526 · AUC = 0,841 · rappel = 0,813 · précision = 0,389 · accuracy = 0,721 |

## Structure du dépôt

```
├── projet_qualite_vin.ipynb   # notebook complet (prétraitement → optimisation), exécuté
├── rapport_technique.md       # rapport technique (version PDF : rapport_technique.pdf)
├── figures/                   # figures générées par le notebook et utilisées dans le rapport
├── data/                      # winequality-red.csv, winequality-white.csv
├── src/vin_qualite.py         # fonctions du projet + réentraînement du modèle final
├── tests/test_vin_qualite.py  # tests unitaires (pytest)
├── app.py                     # application web Streamlit de prédiction
├── Dockerfile                 # conteneur de l'application
├── requirements.txt           # dépendances du notebook et de l'application
└── requirements-app.txt       # dépendances minimales du conteneur
```

## Lancer le projet

```bash
pip install -r requirements.txt

# Notebook
jupyter notebook projet_qualite_vin.ipynb

# Réentraîner et sauvegarder le modèle final (models/modele_final.joblib)
python src/vin_qualite.py

# Tests unitaires
python -m pytest

# Application web (http://localhost:8501)
streamlit run app.py
```

Sous Google Colab : téléverser le notebook et les deux fichiers CSV dans un dossier `data/`.

## Conteneur Docker

```bash
docker build -t qualite-vin .
docker run -p 8501:8501 qualite-vin
```

L'image réentraîne le modèle et lance les tests pendant la construction, puis démarre l'application sur http://localhost:8501.

## Publier sur GitHub / GitLab

Le dépôt Git local est déjà initialisé avec un premier commit. Créer un dépôt vide sur GitHub, puis :

```bash
git remote add origin https://github.com/<utilisateur>/<depot>.git
git push -u origin main
```

## Source des données

P. Cortez, A. Cerdeira, F. Almeida, T. Matos et J. Reis, *Modeling wine preferences by data mining from physicochemical properties*, Decision Support Systems, 47(4), 2009. Jeu *Wine Quality*, UCI Machine Learning Repository (licence CC BY 4.0).
