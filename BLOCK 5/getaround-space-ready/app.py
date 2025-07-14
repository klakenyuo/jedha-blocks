import streamlit as st
import requests

st.set_page_config(page_title="Getaround Dashboard", layout="centered")

st.title("🚗 Getaround Car Price Predictor")
st.markdown("Prédisez le prix d'une voiture en utilisant l'API FastAPI déployée dans ce même espace.")

# Formulaire utilisateur
with st.form("prediction_form"):
    model_key = st.selectbox("Modèle", ["renault", "citroen", "peugeot"])
    mileage = st.number_input("Kilométrage", 0, 300000, step=1000)
    engine_power = st.number_input("Puissance moteur", 50, 300)
    fuel = st.selectbox("Carburant", ["diesel", "petrol", "hybrid", "electric"])
    paint_color = st.selectbox("Couleur", ["grey", "black", "white", "blue", "red"])
    car_type = st.selectbox("Type de voiture", ["sedan", "hatchback", "wagon", "van", "suv"])
    private_parking_available = st.checkbox("Parking privé")
    has_gps = st.checkbox("GPS")
    has_air_conditioning = st.checkbox("Climatisation")
    automatic_car = st.checkbox("Automatique")
    has_getaround_connect = st.checkbox("Getaround Connect")
    has_speed_regulator = st.checkbox("Régulateur de vitesse")
    winter_tires = st.checkbox("Pneus hiver")
    
    submitted = st.form_submit_button("Prédire")

if submitted:
    payload = {
        "model_key": model_key,
        "mileage": mileage,
        "engine_power": engine_power,
        "fuel": fuel,
        "paint_color": paint_color,
        "car_type": car_type,
        "private_parking_available": private_parking_available,
        "has_gps": has_gps,
        "has_air_conditioning": has_air_conditioning,
        "automatic_car": automatic_car,
        "has_getaround_connect": has_getaround_connect,
        "has_speed_regulator": has_speed_regulator,
        "winter_tires": winter_tires
    }

    try:
        response = requests.post("http://localhost:7860/predict", json=payload)
        if response.status_code == 200:
            result = response.json()
            st.success(f"💰 Prix estimé : **{result['prix']} €** (confiance : {result['confidence']})")
        else:
            st.error("Erreur lors de la prédiction. Vérifiez les valeurs envoyées.")
    except Exception as e:
        st.error(f"Erreur lors de la connexion à l'API : {e}")
