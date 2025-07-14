# Prédiction des Ventes Walmart

Ce projet utilise le machine learning pour prédire les ventes hebdomadaires des magasins Walmart en se basant sur diverses variables explicatives.

## Description du Projet

Le projet est divisé en trois parties principales :

1. **Exploratory Data Analysis (EDA)**
   - Analyse des données brutes
   - Gestion des valeurs manquantes
   - Visualisation des distributions et des relations entre variables
   - Prétraitement des données

2. **Modèle de Base (Régression Linéaire)**
   - Implémentation d'un modèle de régression linéaire simple
   - Évaluation des performances sur les ensembles d'entraînement et de test
   - Analyse des coefficients pour comprendre l'importance des features

3. **Lutte contre le Surapprentissage**
   - Implémentation de modèles régularisés (Ridge et Lasso)
   - Optimisation des hyperparamètres avec GridSearchCV
   - Comparaison des performances entre les différents modèles
   - Analyse des courbes d'apprentissage

## Structure des Données

Le dataset contient les variables suivantes :
- `Store` : Identifiant du magasin
- `Date` : Date de la vente
- `Weekly_Sales` : Ventes hebdomadaires (variable cible)
- `Holiday_Flag` : Indicateur de jour férié
- `Temperature` : Température
- `Fuel_Price` : Prix du carburant
- `CPI` : Indice des prix à la consommation
- `Unemployment` : Taux de chômage

## Technologies Utilisées

- Python 3.x
- Pandas pour la manipulation des données
- NumPy pour les calculs numériques
- Scikit-learn pour le machine learning
- Matplotlib et Seaborn pour la visualisation

## Installation

1. Clonez le repository
2. Installez les dépendances requises :
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

## Utilisation

1. Assurez-vous que le fichier `dataset.csv` est présent dans le répertoire du projet
2. Ouvrez le notebook `main.ipynb` avec Jupyter Notebook ou JupyterLab
3. Exécutez les cellules dans l'ordre

## Structure du Projet

```
block_4_ml_wallmart/
├── main.ipynb          # Notebook principal contenant l'analyse et les modèles
├── dataset.csv         # Données d'entrée
└── README.md           # Ce fichier
```

## Résultats

Le projet compare les performances de trois modèles :
1. Régression Linéaire simple
2. Régression Ridge
3. Régression Lasso

Les performances sont évaluées à l'aide de :
- RMSE (Root Mean Square Error)
- R² (coefficient de détermination)
- Courbes d'apprentissage

## Auteur

Gilles Akakpo 