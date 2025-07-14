# Projet de Prédiction de Prix

Ce projet consiste en un système de prédiction de prix avec une API REST et un dashboard interactif. Le système utilise des modèles de machine learning pour prédire les prix et offre une interface utilisateur pour visualiser les résultats.

## Structure du Projet

```
.
├── api/                    # API REST Flask
│   ├── app.py             # Application principale de l'API
│   └── requirements.txt    # Dépendances de l'API
├── dashboard/             # Interface utilisateur Streamlit
│   └── app.py            # Application du dashboard
├── data/                  # Données d'entraînement et d'analyse
│   ├── delay_analysis.csv
│   ├── delay_analysis.xlsx
│   └── pricing_project.csv
└── model/                 # Modèles entraînés
    ├── finalized_model.sav
    ├── finalized_prepoc.sav
    └── notebook.ipynb
```

## Fonctionnalités

- **API REST** : Service de prédiction accessible via des endpoints REST
- **Dashboard** : Interface utilisateur interactive pour visualiser les prédictions
- **Modèle de ML** : Système de prédiction basé sur des données historiques
- **Analyse de Délais** : Outils d'analyse des délais de livraison

## Installation

1. Cloner le repository :
```bash
git clone [URL_DU_REPO]
cd [NOM_DU_PROJET]
```

2. Installer les dépendances :
```bash
pip install -r api/requirements.txt
```

## Utilisation

### API

Pour démarrer l'API :
```bash
cd api
python app.py
```

L'API sera accessible à l'adresse : `http://localhost:5000`

### Dashboard

Pour lancer le dashboard :
```bash
cd dashboard
streamlit run app.py
```

Le dashboard sera accessible à l'adresse : `http://localhost:8501`

### MLflow

Pour lancer MLflow et accéder à l'interface de suivi des expériences :
```bash
cd model
mlflow ui
```

L'interface MLflow sera accessible à l'adresse : `http://localhost:5000`

## Technologies Utilisées

- Python
- Flask (API)
- Streamlit (Dashboard)
- Pandas (Traitement des données)
- Scikit-learn (Machine Learning)
- Plotly (Visualisation)
- Matplotlib/Seaborn (Visualisation)

## Auteurs

Gilles AKAKPO
 