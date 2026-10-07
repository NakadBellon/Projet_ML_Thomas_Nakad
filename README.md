# Classification de la qualité du vin

Projet 2 – Machine Learning I – ING2-BDML – EFREI 2026/2027

Prédire si un vin rouge est **bon (note ≥ 7)** à partir de ses mesures physico-chimiques.

## Résultats

| | |
|---|---|
| Données | 1 599 vins rouges, 11 variables, 13,6 % de bons vins |
| Sélection de variables | 8 variables retenues (filter, wrapper, embedded) |
| Modèles comparés | Arbre de décision, Naive Bayes, régression logistique, k-NN (k = 1, 3, 5, 7), SVM (noyaux linéaire, RBF, sigmoïde, polynomial) |
| Optimisation | Grid Search et Random Search, validation croisée à 5 plis, critère : moyenne accuracy + rappel |
| **Modèle retenu** | **Régression logistique** (`C=10`, `class_weight='balanced'`) |
| Test (400 vins) | Moyenne accuracy + rappel = 0,812 · rappel = 0,844 · précision = 0,319 · AUC = 0,869 |

## Structure du dépôt

```
├── projet_qualite_vin.ipynb   # notebook complet, exécuté
├── rapport_technique.md       # rapport technique (version PDF : rapport_technique.pdf)
├── figures/                   # figures du notebook utilisées dans le rapport
├── data/winequality-red.csv   # données
├── src/vin_qualite.py         # fonctions du projet + réentraînement du modèle final
├── tests/test_vin_qualite.py  # tests unitaires (pytest)
├── app.py                     # application web Streamlit
├── Dockerfile                 # conteneur de l'application
├── requirements.txt           # dépendances du notebook et de l'application
└── requirements-app.txt       # dépendances du conteneur
```

## Lancer le projet

```bash
pip install -r requirements.txt

jupyter notebook projet_qualite_vin.ipynb   # notebook
python src/vin_qualite.py                   # réentraîner et sauvegarder le modèle final
python -m pytest                            # tests unitaires
streamlit run app.py                        # application web (http://localhost:8501)
```

Sous Google Colab : téléverser le notebook et `winequality-red.csv` dans un dossier `data/`.

## Conteneur Docker

```bash
docker build -t qualite-vin .
docker run -p 8501:8501 qualite-vin
```

## Publier sur GitHub / GitLab

Créer un dépôt vide, puis :

```bash
git remote add origin https://github.com/<utilisateur>/<depot>.git
git push -u origin main
```

## Source des données

P. Cortez, A. Cerdeira, F. Almeida, T. Matos et J. Reis, *Modeling wine preferences by data mining from physicochemical properties*, Decision Support Systems, 47(4), 2009.
