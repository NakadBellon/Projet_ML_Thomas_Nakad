"""Fonctions du projet « Classification de la qualité du vin ».

Ce module reprend les étapes du notebook projet_qualite_vin.ipynb pour les bonus
(tests unitaires et application web). Lancé directement (python src/vin_qualite.py),
il réentraîne le modèle final et le sauvegarde dans models/modele_final.joblib.
"""
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FICHIER_DONNEES = os.path.join(RACINE, 'data', 'winequality-red.csv')
CHEMIN_MODELE = os.path.join(RACINE, 'models', 'modele_final.joblib')

VARIABLES = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
             'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
             'pH', 'sulphates', 'alcohol']

# Résultats du notebook : variables retenues par la sélection (section 1.11)
# et meilleurs hyperparamètres du modèle final (section 4)
VARIABLES_SELECTIONNEES = ['fixed acidity', 'volatile acidity', 'citric acid', 'chlorides',
                           'total sulfur dioxide', 'density', 'sulphates', 'alcohol']
PARAMS_FINAUX = {'C': 10, 'class_weight': 'balanced'}


def charger_donnees(fichier=FICHIER_DONNEES):
    """Charge le fichier des vins rouges."""
    return pd.read_csv(fichier, sep=';', header=0)


def creer_cible(vin):
    """Cible binaire du sujet : bon vin si qualité >= 7."""
    vin = vin.copy()
    vin['bon_vin'] = (vin['quality'] >= 7).astype(int)
    return vin.drop(columns='quality')


def preparer_donnees(fichier=FICHIER_DONNEES):
    """Retourne X (les 11 variables) et Y (bon_vin) en tableaux numpy."""
    vin = creer_cible(charger_donnees(fichier))
    return vin[VARIABLES].values, vin['bon_vin'].values


def separer(X, Y):
    """Séparation du TP1 : 25 % de test, random_state=1."""
    return train_test_split(X, Y, test_size=0.25, random_state=1)


def selectionner(X):
    """Garde uniquement les variables retenues par la sélection de variables."""
    idx = [VARIABLES.index(v) for v in VARIABLES_SELECTIONNEES]
    return X[:, idx]


def moyenne_acc_rappel(y_vrai, y_pred):
    """Critère du TP1 : moyenne de l'accuracy et du rappel."""
    return (accuracy_score(y_vrai, y_pred) + recall_score(y_vrai, y_pred)) / 2


def evaluer_modele(modele, Xte, Yte):
    """Métriques du sujet pour un modèle déjà entraîné (données déjà normalisées)."""
    Ypred = modele.predict(Xte)
    return {'Accuracy': accuracy_score(Yte, Ypred),
            'Précision': precision_score(Yte, Ypred, zero_division=0),
            'Rappel': recall_score(Yte, Ypred),
            'F1': f1_score(Yte, Ypred),
            'AUC': roc_auc_score(Yte, modele.predict_proba(Xte)[:, 1]),
            'Moy. acc+rappel (TP1)': moyenne_acc_rappel(Yte, Ypred)}


def entrainer_et_sauvegarder(fichier=FICHIER_DONNEES, chemin=CHEMIN_MODELE):
    """Réentraîne le modèle final (normalisation + régression logistique) et le sauvegarde."""
    X, Y = preparer_donnees(fichier)
    Xtrain, Xtest, Ytrain, Ytest = separer(X, Y)
    Xtrain_sel, Xtest_sel = selectionner(Xtrain), selectionner(Xtest)
    ss = StandardScaler()
    ss.fit(Xtrain_sel)
    modele = LogisticRegression(max_iter=1000, **PARAMS_FINAUX)
    modele.fit(ss.transform(Xtrain_sel), Ytrain)
    os.makedirs(os.path.dirname(chemin), exist_ok=True)
    objet = {'normalisation': ss, 'modele': modele, 'variables': VARIABLES_SELECTIONNEES,
             'nom': 'Régression logistique'}
    joblib.dump(objet, chemin)
    return objet, evaluer_modele(modele, ss.transform(Xtest_sel), Ytest)


def charger_modele(chemin=CHEMIN_MODELE):
    """Charge le modèle sauvegardé ; le réentraîne s'il est absent ou illisible."""
    try:
        return joblib.load(chemin)
    except Exception:
        objet, _ = entrainer_et_sauvegarder(chemin=chemin)
        return objet


def predire(objet, mesures):
    """Prédiction pour un vin décrit par un dictionnaire {variable: valeur}.
    Retourne 1 (bon vin) ou 0 (pas bon)."""
    x = np.array([[mesures[v] for v in objet['variables']]], dtype=float)
    return int(objet['modele'].predict(objet['normalisation'].transform(x))[0])


if __name__ == '__main__':
    _, metriques = entrainer_et_sauvegarder()
    print('Modèle final sauvegardé dans', CHEMIN_MODELE)
    for nom, valeur in metriques.items():
        print('{0:<25} {1:.3f}'.format(nom, valeur))
