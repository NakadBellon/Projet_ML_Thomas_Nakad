"""Application web : prédiction de la qualité d'un vin (bonus du projet).

Lancement : streamlit run app.py
"""
import os
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
import vin_qualite as vq  # noqa: E402

st.set_page_config(page_title='Qualité du vin', page_icon='🍷', layout='centered')

LIBELLES = {
    'fixed acidity': ('Acidité fixe (g/L)', 3.8, 15.9, 0.1),
    'volatile acidity': ('Acidité volatile (g/L)', 0.08, 1.6, 0.01),
    'citric acid': ('Acide citrique (g/L)', 0.0, 1.7, 0.01),
    'residual sugar': ('Sucre résiduel (g/L)', 0.6, 66.0, 0.1),
    'chlorides': ('Chlorures (g/L)', 0.009, 0.62, 0.001),
    'free sulfur dioxide': ('SO₂ libre (mg/L)', 1.0, 290.0, 1.0),
    'total sulfur dioxide': ('SO₂ total (mg/L)', 6.0, 440.0, 1.0),
    'density': ('Densité (g/cm³)', 0.9870, 1.0390, 0.0001),
    'pH': ('pH', 2.7, 4.1, 0.01),
    'sulphates': ('Sulfates (g/L)', 0.22, 2.0, 0.01),
    'alcohol': ('Alcool (% vol.)', 8.0, 15.0, 0.1),
}


@st.cache_resource
def obtenir_modele():
    return vq.charger_modele()


@st.cache_data
def obtenir_exemples():
    """Jeu de test du notebook (mêmes prétraitements, même séparation)."""
    X, Y = vq.preparer_donnees()
    _, Xtest, _, Ytest = vq.separer(X, Y)
    exemples = pd.DataFrame(Xtest, columns=vq.VARIABLES)
    exemples['bon_vin'] = Ytest
    return exemples


modele = obtenir_modele()
exemples = obtenir_exemples()

st.title('🍷 Prédiction de la qualité d\'un vin')
st.write('Modèle : **SVM à noyau RBF** optimisé (C = 100, gamma = 0,01, classes équilibrées), '
         'entraîné sur le jeu *Wine Quality*. Un vin est considéré comme **bon** si sa note est **≥ 7**.')

st.subheader('1. Choisir un vin')
mode = st.radio('Source des mesures', ['Un vin du jeu de test', 'Saisie manuelle'], horizontal=True)

if mode == 'Un vin du jeu de test':
    numero = st.number_input('Numéro du vin dans le jeu de test', min_value=0,
                             max_value=len(exemples) - 1, value=0, step=1)
    ligne = exemples.iloc[int(numero)]
    mesures = {v: float(ligne[v]) for v in vq.VARIABLES}
    verite = int(ligne['bon_vin'])
    st.dataframe(pd.DataFrame([mesures]).rename(columns={k: v[0] for k, v in LIBELLES.items()}),
                 hide_index=True)
else:
    verite = None
    mesures = {}
    type_vin = st.selectbox('Type de vin', ['blanc', 'rouge'])
    mesures['type_rouge'] = 1 if type_vin == 'rouge' else 0
    mediane = exemples[exemples['type_rouge'] == mesures['type_rouge']].median()
    colonnes = st.columns(2)
    for i, (variable, (libelle, mini, maxi, pas)) in enumerate(LIBELLES.items()):
        with colonnes[i % 2]:
            mesures[variable] = st.number_input(libelle, min_value=float(mini), max_value=float(maxi),
                                                value=float(min(max(mediane[variable], mini), maxi)),
                                                step=float(pas), format='%.4f' if pas < 0.001 else '%.3f')

st.subheader('2. Prédiction')
classe, proba = vq.predire(modele, mesures)
if classe == 1:
    st.success('Prédiction : **bon vin** (qualité ≥ 7)')
else:
    st.error('Prédiction : **vin non classé « bon »** (qualité < 7)')
seuil = vq.seuil_score(modele)
st.metric('Probabilité estimée d\'être un bon vin', '{:.0%}'.format(proba),
          delta='seuil de décision : {:.0%}'.format(seuil), delta_color='off')
st.progress(min(proba, 1.0))
st.caption('Le modèle est entraîné avec des classes équilibrées (class_weight=\'balanced\') : il privilégie '
           'le rappel et classe un vin « bon » dès que sa probabilité dépasse {:.0%}, et non 50 %. '
           'Sur le jeu de test, il retrouve 81 % des bons vins, mais 39 % seulement de ses alertes '
           'sont de vrais bons vins.'.format(seuil))

if verite is not None:
    if verite == classe:
        st.info('Valeur réelle : **{}** → prédiction correcte ✅'.format('bon vin' if verite else 'pas bon'))
    else:
        st.warning('Valeur réelle : **{}** → prédiction incorrecte ❌'.format('bon vin' if verite else 'pas bon'))
