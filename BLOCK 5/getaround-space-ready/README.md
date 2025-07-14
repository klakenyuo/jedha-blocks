# Getaround Pricing & Delay Prediction

Ce projet regroupe deux modules principaux :
- 🔮 Une **API FastAPI** pour la prédiction du prix d'une voiture en fonction de ses caractéristiques.
- 📊 Un **Dashboard Streamlit** pour visualiser les retards de retour de véhicule et explorer les données.

---

## 🚀 Lancer l'application avec Docker

### 1. Prérequis
- Docker et Docker Compose installés

### 2. Lancer l'application

```bash
docker-compose up --build
```

---

## 📂 Architecture du projet

```
getaround-docker/
├── api/                 # API FastAPI (endpoint /predict)
├── dashboard/           # Application Streamlit
├── model/               # Modèles entraînés
├── data/                # Données Excel et CSV
├── Dockerfile.api       # Image de l'API
├── Dockerfile.dashboard # Image du dashboard
└── docker-compose.yml   # Orchestration Docker
```

---

## 🌐 Accès aux interfaces

- 📊 Dashboard : [http://localhost:8501](http://localhost:8501)
- 🧠 API Swagger : [http://localhost:7860/docs](http://localhost:7860/docs)

---

## 📁 Dossier `data/`

Ce dossier contient :
- `delay_analysis.csv` et `.xlsx` : données d’analyse de retards
- `pricing_project.csv` : dataset utilisé pour entraîner le modèle de prédiction

---

## 🧠 À propos de l'API

**Endpoint principal :**
```
POST /predict
```

**Exemple de payload JSON :**
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

---

## 👤 Auteur
**Gilles AKAKPO**  
Projet réalisé dans le cadre d’une étude de cas sur la plateforme Getaround.
