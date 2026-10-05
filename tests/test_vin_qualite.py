"""Tests unitaires du projet (lancer avec : python -m pytest)."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
import vin_qualite as vq  # noqa: E402


@pytest.fixture(scope='module')
def vin_brut():
    return vq.charger_donnees()


@pytest.fixture(scope='module')
def donnees():
    X, Y = vq.preparer_donnees()
    return vq.separer(X, Y)


@pytest.fixture(scope='module')
def modele(donnees):
    Xtrain, Xtest, Ytrain, Ytest = donnees
    return vq.construire_modele_final().fit(Xtrain, Ytrain)


def petit_jeu():
    """Petit jeu synthétique : 2 vins identiques, 1 vin très extrême, notes 5 à 8."""
    lignes = []
    for i in range(30):
        ligne = {v: 1.0 + 0.01 * i for v in vq.VARIABLES_NUM}
        ligne['quality'] = 5 + (i % 4)
        ligne['type'] = 'rouge' if i % 2 else 'blanc'
        lignes.append(ligne)
    lignes.append(dict(lignes[0]))                  # doublon
    extreme = dict(lignes[1])
    extreme['chlorides'] = 1000.0                   # valeur aberrante extrême
    lignes.append(extreme)
    return pd.DataFrame(lignes)


# ----- Chargement
def test_chargement_taille(vin_brut):
    assert vin_brut.shape == (6497, 13)
    assert set(vin_brut['type']) == {'rouge', 'blanc'}


def test_aucune_valeur_manquante(vin_brut):
    assert vin_brut.isnull().sum().sum() == 0


# ----- Nettoyage
def test_nettoyer_supprime_doublons_et_extremes():
    vin = vq.nettoyer(petit_jeu())
    assert vin.duplicated().sum() == 0
    assert vin['chlorides'].max() < 1000
    assert len(vin) == 30


def test_nettoyer_jeu_reel(vin_brut):
    vin = vq.nettoyer(vin_brut)
    assert vin.duplicated().sum() == 0
    assert len(vin) == 5258


# ----- Encodage et cible
def test_encodage_type_et_cible():
    vin = vq.encoder_et_creer_cible(petit_jeu())
    assert 'type' not in vin.columns and 'quality' not in vin.columns
    assert set(vin['type_rouge']) == {0, 1}
    # qualité >= 7 -> bon vin
    attendu = (petit_jeu()['quality'] >= 7).astype(int).values
    assert (vin['bon_vin'].values == attendu).all()


def test_cible_binaire_et_desequilibree():
    X, Y = vq.preparer_donnees()
    assert set(np.unique(Y)) == {0, 1}
    assert X.shape[1] == len(vq.VARIABLES)
    assert 0.15 < Y.mean() < 0.25


# ----- Séparation
def test_separation_stratifiee(donnees):
    Xtrain, Xtest, Ytrain, Ytest = donnees
    total = len(Ytrain) + len(Ytest)
    assert abs(len(Ytest) / total - 0.25) < 0.01
    assert abs(Ytrain.mean() - Ytest.mean()) < 0.01


# ----- Évaluation
def test_evaluer_modele_bornes(modele, donnees):
    _, Xtest, _, Ytest = donnees
    metriques = vq.evaluer_modele(modele, Xtest, Ytest)
    assert set(metriques) == {'Accuracy', 'Précision', 'Rappel', 'F1', 'AUC', 'Moy. acc+rappel (TP1)'}
    for valeur in metriques.values():
        assert 0.0 <= valeur <= 1.0


def test_modele_final_meilleur_que_le_hasard(modele, donnees):
    _, Xtest, _, Ytest = donnees
    metriques = vq.evaluer_modele(modele, Xtest, Ytest)
    assert metriques['AUC'] > 0.8
    assert metriques['F1'] > 0.45


# ----- Prédiction
def test_predire_un_vin(modele):
    vin_alcoolise = {'fixed acidity': 6.8, 'volatile acidity': 0.25, 'citric acid': 0.34,
                     'residual sugar': 2.0, 'chlorides': 0.03, 'free sulfur dioxide': 30,
                     'total sulfur dioxide': 110, 'density': 0.9900, 'pH': 3.2,
                     'sulphates': 0.5, 'alcohol': 12.8, 'type_rouge': 0}
    classe, proba = vq.predire(modele, vin_alcoolise)
    assert classe in (0, 1)
    assert 0.0 <= proba <= 1.0


def test_seuil_score_coherent_avec_predict(modele, donnees):
    _, Xtest, _, _ = donnees
    seuil = vq.seuil_score(modele)
    proba = modele.predict_proba(Xtest)[:, 1]
    classe = modele.predict(Xtest)
    assert 0.0 < seuil < 0.5
    assert proba[classe == 1].min() >= seuil - 0.01
    assert proba[classe == 0].max() <= seuil + 0.01


def test_predire_variable_manquante(modele):
    with pytest.raises(KeyError):
        vq.predire(modele, {'alcohol': 12.0})
