"""Fonctions du projet « Classification de la qualité du vin ».

Ce module reprend, sous forme de fonctions réutilisables, les étapes du notebook
projet_qualite_vin.ipynb. Il sert aux tests unitaires et à l'application web.
Lancé directement (python src/vin_qualite.py), il réentraîne le modèle final
et le sauvegarde dans models/modele_final.joblib.
"""
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOSSIER_DONNEES = os.path.join(RACINE, 'data')
CHEMIN_MODELE = os.path.join(RACINE, 'models', 'modele_final.joblib')

VARIABLES_NUM = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
                 'pH', 'sulphates', 'alcohol']
VARIABLES = VARIABLES_NUM + ['type_rouge']
SEUIL_BON_VIN = 7
SEUIL_ZSCORE = 5

# Meilleurs hyperparamètres trouvés par GridSearchCV dans le notebook (section 4)
PARAMS_FINAUX = {'C': 100, 'gamma': 0.01, 'class_weight': 'balanced'}


def charger_donnees(dossier=DOSSIER_DONNEES):
    """Charge les vins rouges et blancs et ajoute la colonne 'type'."""
    rouge = pd.read_csv(os.path.join(dossier, 'winequality-red.csv'), sep=';', header=0)
    blanc = pd.read_csv(os.path.join(dossier, 'winequality-white.csv'), sep=';', header=0)
    rouge['type'] = 'rouge'
    blanc['type'] = 'blanc'
    return pd.concat([rouge, blanc], ignore_index=True)


def nettoyer(vin, seuil_z=SEUIL_ZSCORE):
    """Supprime les doublons puis les vins ayant une valeur à plus de seuil_z écarts-types."""
    vin = vin.drop_duplicates().reset_index(drop=True)
    z = (vin[VARIABLES_NUM] - vin[VARIABLES_NUM].mean()) / vin[VARIABLES_NUM].std()
    extremes = (z.abs() > seuil_z).any(axis=1)
    return vin[~extremes].reset_index(drop=True)


def encoder_et_creer_cible(vin, seuil=SEUIL_BON_VIN):
    """Encode le type (type_rouge = 1 pour rouge) et crée la cible bon_vin (qualité >= seuil)."""
    vin = vin.copy()
    vin['type_rouge'] = (vin['type'] == 'rouge').astype(int)
    vin['bon_vin'] = (vin['quality'] >= seuil).astype(int)
    return vin.drop(columns=['type', 'quality'])


def preparer_donnees(dossier=DOSSIER_DONNEES):
    """Chargement + nettoyage + encodage. Retourne X (numpy), Y (numpy)."""
    vin = encoder_et_creer_cible(nettoyer(charger_donnees(dossier)))
    return vin[VARIABLES].values, vin['bon_vin'].values


def separer(X, Y):
    """Séparation apprentissage / test du TP1 (25 % de test, random_state=1), stratifiée."""
    return train_test_split(X, Y, test_size=0.25, random_state=1, stratify=Y)


def evaluer_modele(modele, Xte, Yte):
    """Retourne les métriques du sujet pour un modèle déjà entraîné."""
    Ypred = modele.predict(Xte)
    if hasattr(modele, 'predict_proba'):
        score = modele.predict_proba(Xte)[:, 1]
    else:
        score = modele.decision_function(Xte)
    acc = accuracy_score(Yte, Ypred)
    rec = recall_score(Yte, Ypred)
    return {'Accuracy': acc,
            'Précision': precision_score(Yte, Ypred, zero_division=0),
            'Rappel': rec,
            'F1': f1_score(Yte, Ypred),
            'AUC': roc_auc_score(Yte, score),
            'Moy. acc+rappel (TP1)': (acc + rec) / 2}


def construire_modele_final(params=PARAMS_FINAUX):
    """Pipeline normalisation (StandardScaler) + SVM RBF optimisé."""
    return Pipeline([('normalisation', StandardScaler()),
                     ('modele', SVC(kernel='rbf', probability=True, random_state=0, **params))])


def entrainer_et_sauvegarder(dossier=DOSSIER_DONNEES, chemin=CHEMIN_MODELE):
    """Réentraîne le modèle final sur la base d'apprentissage et le sauvegarde."""
    X, Y = preparer_donnees(dossier)
    Xtrain, Xtest, Ytrain, Ytest = separer(X, Y)
    modele = construire_modele_final().fit(Xtrain, Ytrain)
    os.makedirs(os.path.dirname(chemin), exist_ok=True)
    joblib.dump({'modele': modele, 'variables': VARIABLES, 'nom': 'SVM RBF'}, chemin)
    return modele, evaluer_modele(modele, Xtest, Ytest)


def charger_modele(chemin=CHEMIN_MODELE):
    """Charge le modèle sauvegardé ; le réentraîne s'il est absent ou illisible
    (par exemple avec une autre version de scikit-learn)."""
    try:
        return joblib.load(chemin)['modele']
    except Exception:
        modele, _ = entrainer_et_sauvegarder(chemin=chemin)
        return modele


def seuil_score(modele):
    """Probabilité « bon vin » au-delà de laquelle le SVM prédit la classe 1.

    Le SVM décide avec sa fonction de décision (> 0 -> classe 1). Ses probabilités
    viennent d'une calibration de Platt : P(classe 0 | f) = 1 / (1 + exp(A f + B)).
    Pour f = 0, P(classe 1) = 1 - 1 / (1 + exp(B)). Avec class_weight='balanced',
    ce seuil est bien inférieur à 0,5 : le modèle privilégie le rappel.
    """
    svc = modele.named_steps['modele']
    return float(1 - 1 / (1 + np.exp(svc.probB_[0])))


def predire(modele, mesures):
    """Prédit pour un vin décrit par un dictionnaire {variable: valeur}.
    Retourne (classe, probabilité d'être un bon vin)."""
    x = np.array([[mesures[v] for v in VARIABLES]], dtype=float)
    proba = float(modele.predict_proba(x)[0, 1])
    return int(modele.predict(x)[0]), proba


if __name__ == '__main__':
    _, metriques = entrainer_et_sauvegarder()
    print('Modèle final sauvegardé dans', CHEMIN_MODELE)
    for nom, valeur in metriques.items():
        print('{0:<25} {1:.3f}'.format(nom, valeur))
