"""Application web : prédiction de la qualité d'un vin.

Lancement : streamlit run app.py
"""
import os
import sys

import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
import vin_qualite as vq  # noqa: E402

st.set_page_config(page_title='Qualité du vin', page_icon='🍷')

# Libellé, minimum, maximum, pas et valeur par défaut (médiane du jeu de données)
CHAMPS = {
    'fixed acidity': ('Acidité fixe (g/L)', 4.6, 15.9, 0.1, 7.9),
    'volatile acidity': ('Acidité volatile (g/L)', 0.12, 1.58, 0.01, 0.52),
    'citric acid': ('Acide citrique (g/L)', 0.0, 1.0, 0.01, 0.26),
    'chlorides': ('Chlorures (g/L)', 0.012, 0.611, 0.001, 0.079),
    'total sulfur dioxide': ('SO₂ total (mg/L)', 6.0, 289.0, 1.0, 38.0),
    'density': ('Densité (g/cm³)', 0.9900, 1.0040, 0.0001, 0.9968),
    'sulphates': ('Sulfates (g/L)', 0.33, 2.0, 0.01, 0.62),
    'alcohol': ('Alcool (% vol.)', 8.4, 14.9, 0.1, 10.2),
}


@st.cache_resource
def obtenir_modele():
    return vq.charger_modele()


objet = obtenir_modele()

st.title('🍷 Prédiction de la qualité d\'un vin rouge')
st.write('Saisissez les mesures du vin. Le modèle (**régression logistique**) indique si le vin est '
         '**bon** (note ≥ 7) ou non.')

mesures = {}
colonnes = st.columns(2)
for i, variable in enumerate(objet['variables']):
    libelle, mini, maxi, pas, defaut = CHAMPS[variable]
    with colonnes[i % 2]:
        mesures[variable] = st.number_input(libelle, min_value=mini, max_value=maxi, value=defaut,
                                            step=pas, format='%.4f' if pas < 0.001 else '%.3f')

if vq.predire(objet, mesures) == 1:
    st.success('Prédiction : **bon vin** (qualité ≥ 7)')
else:
    st.error('Prédiction : **pas bon** (qualité < 7)')
