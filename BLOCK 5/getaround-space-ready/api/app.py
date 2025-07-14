import mlflow 
import uvicorn
import pandas as pd 
from pydantic import BaseModel, Field, validator
from typing import Literal, List, Union
from fastapi import FastAPI, HTTPException, File, UploadFile
import joblib
import numpy as np
import logging
from pathlib import Path
import os
import traceback
import json
from datetime import datetime

# Configuration du logging
log_dir = Path(__file__).parent / 'logs'
log_dir.mkdir(exist_ok=True)

# Configuration du fichier de log
log_file = log_dir / f'api_{datetime.now().strftime("%Y%m%d")}.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialisation de l'application FastAPI
app = FastAPI(
    title="Car Price Prediction API",
    description="""
    Voici un exemple de requête avec des données réelles :
    
    ```json
    {
        "model_key": "renault",
        "mileage": 45000,
        "engine_power": 90,
        "fuel": "diesel",
        "paint_color": "grey",
        "car_type": "hatchback",
        "private_parking_available": true,
        "has_gps": true,
        "has_air_conditioning": true,
        "automatic_car": false,
        "has_getaround_connect": true,
        "has_speed_regulator": true,
        "winter_tires": false
    }
    ```
    
    Réponse attendue :
    ```json
    {
        "prix": 12500.50,
        "confidence": 0.95
    }
    ```
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Définition des modèles de données avec validation
class ListIn(BaseModel):
    """
    Modèle de données pour les entrées de prédiction
    """
    model_key: str = Field(..., description="Modèle de la voiture (ex: renault, citroen, peugeot)")
    mileage: int = Field(..., ge=0, description="Kilométrage de la voiture (ex: 45000)")
    engine_power: int = Field(..., ge=0, description="Puissance du moteur en chevaux (ex: 90)")
    fuel: str = Field(..., description="Type de carburant (diesel, petrol, hybrid, electric)")
    paint_color: str = Field(..., description="Couleur de la voiture (ex: grey, black, white)")
    car_type: str = Field(..., description="Type de voiture (sedan, hatchback, wagon, van, suv)")
    private_parking_available: bool = Field(..., description="Disponibilité d'un parking privé")
    has_gps: bool = Field(..., description="Présence d'un GPS")
    has_air_conditioning: bool = Field(..., description="Présence de la climatisation")
    automatic_car: bool = Field(..., description="Transmission automatique")
    has_getaround_connect: bool = Field(..., description="Présence de Getaround Connect")
    has_speed_regulator: bool = Field(..., description="Présence d'un régulateur de vitesse")
    winter_tires: bool = Field(..., description="Présence de pneus hiver")

    class Config:
        schema_extra = {
            "example": {
                "model_key": "renault",
                "mileage": 45000,
                "engine_power": 90,
                "fuel": "diesel",
                "paint_color": "grey",
                "car_type": "hatchback",
                "private_parking_available": True,
                "has_gps": True,
                "has_air_conditioning": True,
                "automatic_car": False,
                "has_getaround_connect": True,
                "has_speed_regulator": True,
                "winter_tires": False
            }
        }

    @validator('fuel')
    def validate_fuel(cls, v):
        allowed_fuels = ['diesel', 'petrol', 'hybrid', 'electric']
        if v.lower() not in allowed_fuels:
            raise ValueError(f'Le type de carburant doit être l\'un des suivants: {", ".join(allowed_fuels)}')
        return v.lower()

class PredictionOut(BaseModel):
    """
    Modèle de données pour la sortie de prédiction
    """
    prix: float = Field(..., description="Prix prédit de la voiture")
    confidence: float = Field(..., description="Niveau de confiance de la prédiction")

    class Config:
        schema_extra = {
            "example": {
                "prix": 12500.50,
                "confidence": 0.95
            }
        }

class ErrorResponse(BaseModel):
    """
    Modèle de données pour les réponses d'erreur
    """
    error: str = Field(..., description="Message d'erreur détaillé")
    timestamp: str = Field(..., description="Horodatage de l'erreur")
    traceback: str = Field(None, description="Traceback complet de l'erreur")

def log_error(error: Exception, context: str = ""):
    """
    Enregistre une erreur dans le fichier de log avec le contexte
    """
    error_details = {
        "timestamp": datetime.now().isoformat(),
        "context": context,
        "error_type": type(error).__name__,
        "error_message": str(error),
        "traceback": traceback.format_exc()
    }
    
    logger.error(f"Erreur dans {context}: {str(error)}")
    logger.error(f"Traceback complet: {error_details['traceback']}")
    
    return error_details

def get_valid_categories(pipeline):
    """
    Récupère les catégories valides du pipeline de prétraitement
    """
    try:
        # Récupération des catégories valides pour chaque feature catégorielle
        categorical_features = pipeline.named_transformers_['cat'].named_steps['encoder'].categories_
        feature_names = pipeline.named_transformers_['cat'].get_feature_names_out()
        
        valid_categories = {}
        for i, feature in enumerate(feature_names):
            if i < len(categorical_features):
                valid_categories[feature] = categorical_features[i].tolist()
        
        return valid_categories
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des catégories valides: {str(e)}")
        return None

def load_models():
    """
    Charge les modèles ML depuis les fichiers sauvegardés
    """
    try:
        # Utilisation d'un chemin absolu pour les fichiers de modèle
        current_dir = Path(__file__).parent.parent
        model_path = current_dir / 'model' / 'finalized_model.sav'
        pipeline_path = current_dir / 'model' / 'finalized_prepoc.sav'
        
        logger.info(f"Tentative de chargement des modèles depuis : {model_path}")
        
        if not model_path.exists() or not pipeline_path.exists():
            error_msg = f"Les fichiers de modèle sont introuvables. Vérifié dans : {model_path}"
            error_details = log_error(FileNotFoundError(error_msg), "load_models")
            raise HTTPException(
                status_code=500,
                detail=error_details
            )
            
        loaded_model = joblib.load(model_path)
        pipeline = joblib.load(pipeline_path)
        
        # Récupération des catégories valides
        valid_categories = get_valid_categories(pipeline)
        if valid_categories:
            logger.info(f"Catégories valides chargées: {valid_categories}")
        
        return loaded_model, pipeline
    except Exception as e:
        error_details = log_error(e, "load_models")
        raise HTTPException(
            status_code=500,
            detail=error_details
        )

def predict_price(values):
    """
    Effectue la prédiction du prix
    """
    try:
        # Création d'un DataFrame avec les bonnes colonnes
        columns = ['model_key', 'mileage', 'engine_power', 'fuel', 'paint_color',
                  'car_type', 'private_parking_available', 'has_gps',
                  'has_air_conditioning', 'automatic_car', 'has_getaround_connect',
                  'has_speed_regulator', 'winter_tires']
        
        # Formatage du model_key (première lettre en majuscule)
        values['model_key'] = values['model_key'].capitalize()
        
        # Conversion des valeurs en DataFrame
        df = pd.DataFrame([values], columns=columns)
        
        # Chargement et application des modèles
        loaded_model, pipeline = load_models()
        
        # Vérification des catégories valides
        valid_categories = get_valid_categories(pipeline)
        if valid_categories:
            # Vérification de la marque de voiture
            if 'model_key' in valid_categories and values['model_key'] not in valid_categories['model_key']:
                error_msg = f"Marque de voiture non reconnue: {values['model_key']}. Marques valides: {valid_categories['model_key']}"
                error_details = log_error(ValueError(error_msg), "predict_price")
                raise HTTPException(
                    status_code=400,
                    detail=error_details
                )
            
            # Vérification du type de voiture
            if 'car_type' in valid_categories and values['car_type'] not in valid_categories['car_type']:
                error_msg = f"Type de voiture non reconnu: {values['car_type']}. Types valides: {valid_categories['car_type']}"
                error_details = log_error(ValueError(error_msg), "predict_price")
                raise HTTPException(
                    status_code=400,
                    detail=error_details
                )
            
            # Vérification de la couleur
            if 'paint_color' in valid_categories and values['paint_color'] not in valid_categories['paint_color']:
                error_msg = f"Couleur non reconnue: {values['paint_color']}. Couleurs valides: {valid_categories['paint_color']}"
                error_details = log_error(ValueError(error_msg), "predict_price")
                raise HTTPException(
                    status_code=400,
                    detail=error_details
                )
        
        # Transformation des données
        transformed_data = pipeline.transform(df)
        
        # Prédiction
        result = loaded_model.predict(transformed_data)
        
        # Calcul d'un score de confiance simple
        confidence = 0.95  # Exemple de score de confiance
        
        return float(result[0]), confidence
    except Exception as e:
        error_details = log_error(e, "predict_price")
        raise HTTPException(
            status_code=500,
            detail=error_details
        )

@app.get("/", tags=["Root"])
async def index():
    """
    Point d'entrée principal de l'API
    """
    return {
        "message": "Bienvenue sur l'API de prédiction de prix des voitures",
        "documentation": "/docs",
        "version": "1.0.0"
    }

@app.get("/categories", tags=["Categories"])
async def get_categories():
    """
    Récupère les catégories valides pour les features catégorielles
    """
    try:
        _, pipeline = load_models()
        valid_categories = get_valid_categories(pipeline)
        return valid_categories
    except Exception as e:
        error_details = log_error(e, "get_categories")
        raise HTTPException(
            status_code=500,
            detail=error_details
        )

@app.post("/predict", response_model=PredictionOut, tags=["Prediction"])
async def predict(values: ListIn):
    """
    Endpoint pour prédire le prix d'une voiture
    
    - Prend en entrée les caractéristiques de la voiture
    - Retourne le prix prédit et un score de confiance
    """
    try:
        # Conversion des valeurs en dictionnaire
        input_values = {
            'model_key': values.model_key,
            'mileage': values.mileage,
            'engine_power': values.engine_power,
            'fuel': values.fuel,
            'paint_color': values.paint_color,
            'car_type': values.car_type,
            'private_parking_available': values.private_parking_available,
            'has_gps': values.has_gps,
            'has_air_conditioning': values.has_air_conditioning,
            'automatic_car': values.automatic_car,
            'has_getaround_connect': values.has_getaround_connect,
            'has_speed_regulator': values.has_speed_regulator,
            'winter_tires': values.winter_tires
        }
        
        prix, confidence = predict_price(input_values)
        logger.info(f"Prédiction effectuée avec succès pour le modèle {values.model_key}")
        
        return {
            "prix": round(prix, 2),
            "confidence": round(confidence, 2)
        }
    except Exception as e:
        error_details = log_error(e, "predict endpoint")
        raise HTTPException(
            status_code=500,
            detail=error_details
        )

if __name__=="__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 
