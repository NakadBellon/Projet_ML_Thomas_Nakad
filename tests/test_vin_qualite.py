"""Tests unitaires du projet (lancer avec : python -m pytest)."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
import vin_qualite as vq  # noqa: E402


@pytest.fixture(scope='module')
def donnees():
    X, Y = vq.preparer_donnees()
    return vq.separer(X, Y)


@pytest.fixture(scope='module')
def modele_et_metriques(tmp_path_factory):
    chemin = str(tmp_path_factory.mktemp('modele') / 'modele.joblib')
    return vq.entrainer_et_sauvegarder(chemin=chemin)


# ----- Chargement et prétraitement
def test_chargement_taille():
    vin = vq.charger_donnees()
    assert vin.shape == (1599, 12)


def test_aucune_valeur_manquante():
    assert vq.charger_donnees().isnull().sum().sum() == 0


def test_cible_qualite_superieure_ou_egale_a_7():
    vin = pd.DataFrame({v: [1.0, 1.0, 1.0] for v in vq.VARIABLES})
    vin['quality'] = [6, 7, 8]
    resultat = vq.creer_cible(vin)
    assert 'quality' not in resultat.columns
    assert list(resultat['bon_vin']) == [0, 1, 1]


def test_cible_desequilibree():
    X, Y = vq.preparer_donnees()
    assert X.shape == (1599, 11)
    assert set(np.unique(Y)) == {0, 1}
    assert abs(Y.mean() - 0.136) < 0.001


# ----- Séparation et sélection
def test_separation_25_pourcent(donnees):
    Xtrain, Xtest, Ytrain, Ytest = donnees
    assert len(Ytrain) == 1199 and len(Ytest) == 400


def test_selection_des_variables(donnees):
    Xtrain = donnees[0]
    assert vq.selectionner(Xtrain).shape == (1199, len(vq.VARIABLES_SELECTIONNEES))


# ----- Modèle final
def test_metriques_entre_0_et_1(modele_et_metriques):
    _, metriques = modele_et_metriques
    for valeur in metriques.values():
        assert 0.0 <= valeur <= 1.0


def test_resultats_du_notebook(modele_et_metriques):
    _, metriques = modele_et_metriques
    assert round(metriques['Moy. acc+rappel'], 3) == 0.812
    assert round(metriques['Rappel'], 3) == 0.844


# ----- Prédiction
def test_predire_un_vin(modele_et_metriques):
    objet, _ = modele_et_metriques
    mesures = {'fixed acidity': 8.0, 'volatile acidity': 0.35, 'citric acid': 0.4,
               'chlorides': 0.07, 'total sulfur dioxide': 30.0, 'density': 0.995,
               'sulphates': 0.8, 'alcohol': 12.5}
    assert vq.predire(objet, mesures) in (0, 1)


def test_predire_variable_manquante(modele_et_metriques):
    objet, _ = modele_et_metriques
    with pytest.raises(KeyError):
        vq.predire(objet, {'alcohol': 12.0})
