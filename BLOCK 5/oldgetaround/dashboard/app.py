import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import joblib
from typing import Dict, List, Any
import logging

# Configuration de la page Streamlit
st.set_page_config(
    page_title='GetAround project',
    page_icon='🚗',
    layout="wide",
    initial_sidebar_state="auto",
    menu_items=None
)

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constantes
DATA_FILES = {
    'pricing': '../data/pricing_project.csv',
    'delay': '../data/delay_analysis.csv'
}

# Fonctions utilitaires
def load_data(file_path: str, sep: str = ',') -> pd.DataFrame:
    """
    Charge les données depuis un fichier CSV.
    
    Args:
        file_path (str): Chemin du fichier CSV
        sep (str): Séparateur utilisé dans le fichier CSV
        
    Returns:
        pd.DataFrame: DataFrame contenant les données
    """
    try:
        return pd.read_csv(file_path, sep=sep)
    except Exception as e:
        logger.error(f"Erreur lors du chargement des données: {e}")
        st.error(f"Erreur lors du chargement des données: {e}")
        return pd.DataFrame()

def display_metrics(df_delay: pd.DataFrame, df_pricing: pd.DataFrame, col: st.columns) -> None:
    """
    Affiche les métriques principales dans les colonnes spécifiées.
    
    Args:
        df_delay (pd.DataFrame): DataFrame contenant les données de retard
        df_pricing (pd.DataFrame): DataFrame contenant les données de prix
        col (st.columns): Colonnes Streamlit pour l'affichage
    """
    nb_rentals = len(df_delay)
    
    with col[0]:
        st.metric(
            label="Nombres de voitures dans le parc :",
            value=df_delay['car_id'].nunique()
        )
        connect_percentage = round(
            len(df_pricing[df_pricing['has_getaround_connect'] == True]) / len(df_pricing) * 100
        )
        st.metric(
            label="Pourcentage de voitures équipées 'Connect' :",
            value=f"{connect_percentage} %"
        )
    
    with col[2]:
        st.metric(
            label="Nombres de locations :",
            value=nb_rentals
        )
        connect_rentals_percentage = round(
            len(df_delay[df_delay['checkin_type'] == 'connect']) / nb_rentals * 100
        )
        st.metric(
            label="Pourcentage de location via 'Connect' :",
            value=f"{connect_rentals_percentage} %"
        )
    
    with col[1]:
        delay_percentage = round(
            len(df_delay[df_delay['delay_at_checkout_in_minutes'] > 0]) / nb_rentals * 100
        )
        st.metric(
            label="Pourcentage de locations rendues avec retard :",
            value=f"{delay_percentage} %"
        )
        cancel_percentage = round(
            len(df_delay[df_delay['state'] == 'canceled']) / nb_rentals * 100
        )
        st.metric(
            label="Pourcentage de locations annulées :",
            value=f"{cancel_percentage} %"
        )

def main_page() -> None:
    """Page d'accueil de l'application."""
    st.markdown("# Accueil")
    st.sidebar.markdown("# Accueil")
    st.header('Statistiques')
    
    # Chargement des données
    dataset_pricing = load_data(DATA_FILES['pricing'])
    dataset_delay = load_data(DATA_FILES['delay'], sep=';')
    
    if dataset_pricing.empty or dataset_delay.empty:
        st.error("Impossible de charger les données. Veuillez vérifier les fichiers.")
        return
    
    # Affichage des métriques
    main_metrics_cols = st.columns([33, 33, 34])
    display_metrics(dataset_delay, dataset_pricing, main_metrics_cols)
    
    # Footer
    st.markdown("---")
    footer = """
        <style>
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: transparent;
            color: white;
            text-align: center;
        }
        </style>
    """
    st.markdown(footer, unsafe_allow_html=True)

def page2() -> None:
    """Page d'analyse des retards."""
    st.title("Dashboard : Analyse d'un jeu de données de GetAround 🚗💲")
    st.markdown("""
        Voici quelques informations clefs pour comprendre la dynamique des retards lors des réservations 
        sur GetAround 🚗, ainsi que leur impact sur les locations, et donc sur le chiffre d'affaire 
        potentiel de GetAround 🚗.
    """)
    st.markdown("---")
    
    # Chargement des données
    dataset_delay = load_data(DATA_FILES['delay'], sep=';')
    if dataset_delay.empty:
        st.error("Impossible de charger les données. Veuillez vérifier les fichiers.")
        return
    
    # Partie 1: Overview des retards
    st.subheader("Partie 1 : Overview des retards")
    main_metrics_cols_1 = st.columns([34, 33, 33])
    
    with main_metrics_cols_1[0]:
        # Graphique des retards
        ended_rentals = dataset_delay[dataset_delay["state"] == "ended"]
        labels = ["A l'heure ou en avance", 'En retard']
        values = [
            len(ended_rentals[ended_rentals["delay_at_checkout_in_minutes"] <= 0]),
            len(ended_rentals[ended_rentals["delay_at_checkout_in_minutes"] > 0])
        ]
        fig = px.pie(
            names=labels,
            values=values,
            title="Part des retards dans les réservations abouties"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with main_metrics_cols_1[1]:
        # Distribution des retards
        delayed_rentals = ended_rentals[ended_rentals["delay_at_checkout_in_minutes"] > 0]
        fig2 = px.histogram(
            delayed_rentals,
            x="delay_at_checkout_in_minutes",
            range_x=[0, 12*60],
            title="Distribution des retards en minutes",
            labels={"delay_at_checkout_in_minutes": "Retard au checkout (mn)"}
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    with main_metrics_cols_1[2]:
        # Métriques des retards
        moyenne_retard = delayed_rentals["delay_at_checkout_in_minutes"].median()
        st.metric(
            label="Retard médian : ",
            value=f"{round(moyenne_retard, 2)} minutes"
        )
        retard_une_h = 100 * (
            len(delayed_rentals[delayed_rentals["delay_at_checkout_in_minutes"] > 60]) /
            len(ended_rentals)
        )
        st.metric(
            label="Retard supérieur à 1h :",
            value=f"{round(retard_une_h, 2)} %"
        )
    
    # Partie 2: Analyse des délais
    st.markdown("---")
    st.subheader("Partie 2 : Impact des délais entre locations")
    main_metrics_cols_2 = st.columns([70, 30])
    
    with main_metrics_cols_2[0]:
        with st.spinner('Chargement...'):
            # Préparation des données
            dataset_delay.dropna(subset=['delay_at_checkout_in_minutes'], inplace=True)
            dataset_delay = dataset_delay.reset_index(drop=True)
            dataset_delay['delay_problem'] = (
                dataset_delay['delay_at_checkout_in_minutes'] -
                dataset_delay['time_delta_with_previous_rental_in_minutes']
            )
            
            # Calcul des statistiques par seuil
            def compute_stats_threshold(delay_tresh: int, check_type: str) -> int:
                mask = (
                    (dataset_delay['delay_problem'] > delay_tresh) &
                    (dataset_delay['checkin_type'] == check_type)
                )
                return dataset_delay[mask].count()[0]
            
            # Calcul des ratios de locations perdues
            nb_rent_connect = dataset_delay[dataset_delay['checkin_type'] == 'connect'].count()[0]
            nb_rent_mobile = dataset_delay[dataset_delay['checkin_type'] == 'mobile'].count()[0]
            
            results = {
                'Threshold (min)': range(400),
                'Rent_lost_mobile(%)': [],
                'Rent_lost_connect(%)': []
            }
            
            for i in range(400):
                results['Rent_lost_mobile(%)'].append(
                    compute_stats_threshold(i, 'mobile') / nb_rent_mobile * 100
                )
                results['Rent_lost_connect(%)'].append(
                    compute_stats_threshold(i, 'connect') / nb_rent_connect * 100
                )
            
            df_delay_stat_treshold = pd.DataFrame(results)
            
            # Affichage du graphique
            st.line_chart(
                data=df_delay_stat_treshold,
                x='Threshold (min)',
                y=["Rent_lost_mobile(%)", 'Rent_lost_connect(%)'],
                use_container_width=True
            )
    
    # Sélecteur de délai
    delay = st.slider(
        'Quel délai en deux locations (en minutes) :',
        0, 400, 60
    )
    
    with main_metrics_cols_2[1]:
        # Affichage des métriques de délai
        st.metric(
            label=f"Pourcentage de location perdue sur mobile pour un délai de {delay} minutes :",
            value=f"{round(df_delay_stat_treshold.iloc[delay][1], 2)} %"
        )
        st.metric(
            label=f"Pourcentage de location perdue sur l'app pour un délai de {delay} minutes :",
            value=f"{round(df_delay_stat_treshold.iloc[delay][2], 2)} %"
        )

def predict_price(values: List[Any]) -> float:
    """
    Prédit le prix de location d'un véhicule.
    
    Args:
        values (List[Any]): Liste des caractéristiques du véhicule
        
    Returns:
        float: Prix prédit
    """
    try:
        predict_array = np.zeros((1, 13))
        im_df = pd.DataFrame(
            predict_array,
            columns=[
                'model_key', 'mileage', 'engine_power', 'fuel', 'paint_color',
                'car_type', 'private_parking_available', 'has_gps',
                'has_air_conditioning', 'automatic_car', 'has_getaround_connect',
                'has_speed_regulator', 'winter_tires'
            ]
        )
        im_df[0:1] = values
        
        loaded_model = joblib.load('../model/finalized_model.sav')
        pipeline = joblib.load('../model/finalized_prepoc.sav')
        
        result = loaded_model.predict(pipeline.transform(im_df))
        return result[0]
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction: {e}")
        st.error(f"Erreur lors de la prédiction: {e}")
        return 0.0

def page3() -> None:
    """Page de prédiction des prix."""
    st.markdown("# Prédiction")
    st.sidebar.markdown("# Prédiction 🎉")
    st.markdown("**Veuillez entrer les informations concernant votre véhicule :**")
    
    # Chargement des données
    dataset_pricing = load_data(DATA_FILES['pricing'])
    if dataset_pricing.empty:
        st.error("Impossible de charger les données. Veuillez vérifier les fichiers.")
        return
    
    # Formulaire de saisie
    col1, col2 = st.columns(2)
    
    with col1:
        marque = st.selectbox(
            'Marque :',
            tuple(dataset_pricing['model_key'].unique())
        )
        kil = st.number_input(
            "Entrer le kilométrage :",
            1, 1000000, 150000, 10
        )
        puissance = st.number_input(
            "Entrer la puissance du véhicule (en CV) :",
            40, 400, 100, 1
        )
        energie = st.selectbox(
            'Carburant :',
            tuple(dataset_pricing['fuel'].unique())
        )
        couleur = st.selectbox(
            'Couleur du véhicule :',
            tuple(dataset_pricing['paint_color'].unique())
        )
        car_type = st.selectbox(
            'Type de véhicule :',
            tuple(dataset_pricing['car_type'].unique())
        )
    
    with col2:
        # Options booléennes
        options = {
            'parking': 'Place de parking privée',
            'gps': 'GPS intégré',
            'ac': 'Climatisation',
            'auto': 'Boîte automatique',
            'gac': 'GetAround Connect',
            'speed': 'Régulateur de vitesse',
            'hiver': 'Pneus hiver'
        }
        
        values = {}
        for key, label in options.items():
            values[key] = st.selectbox(label + ' :', ('Yes', 'No')) == 'Yes'
    
    # Bouton de prédiction
    if st.button("Predict"):
        list_values = [
            marque, int(kil), int(puissance), energie, couleur, car_type,
            values['parking'], values['gps'], values['ac'], values['auto'],
            values['gac'], values['speed'], values['hiver']
        ]
        
        result = predict_price(list_values)
        if result > 0:
            st.success(
                f"Le montant de location à la journée de votre véhicule "
                f"s'élève à {result:.2f} €"
            )

# Configuration des pages
page_names_to_funcs = {
    "Accueil": main_page,
    "Dashboard": page2,
    "Prédiction": page3,
}

# Sélection et affichage de la page
selected_page = st.sidebar.selectbox(
    "Selectionner une page :",
    page_names_to_funcs.keys()
)
page_names_to_funcs[selected_page]()