# Prédiction du Taux de Conversion - Projet Data Science

## Description du Projet

Ce projet vise à prédire la probabilité de conversion des utilisateurs d'une newsletter en utilisant des techniques de machine learning. Le projet est basé sur les données de www.datascienceweekly.org, une newsletter réputée dans le domaine de la data science.

L'objectif principal est de construire un modèle qui permet de prédire si un utilisateur va s'abonner à la newsletter en se basant sur quelques informations sur l'utilisateur. Cette prédiction permettra d'analyser les paramètres du modèle pour identifier les facteurs clés qui influencent la conversion et potentiellement découvrir de nouveaux leviers d'action pour améliorer le taux de conversion.

## Structure du Projet

```
conversion_rate/
├── main.ipynb             # Notebook principal avec l'analyse et le modèle
├── description.ipynb      # Description détaillée du projet
├── conversion_data_train.csv  # Données d'entraînement
├── conversion_data_test.csv   # Données de test
├── conversion_prediction.csv  # Prédictions générées
└── README.md              # Documentation du projet
```

## Objectifs du Projet

1. **Analyse Exploratoire des Données (EDA)**
   - Exploration des données d'entraînement
   - Visualisation des distributions et corrélations
   - Identification des patterns intéressants

2. **Prétraitement et Modélisation**
   - Nettoyage et préparation des données
   - Création d'un modèle de base
   - Évaluation des performances avec le F1-score

3. **Amélioration du Modèle**
   - Feature engineering
   - Sélection des features
   - Optimisation des hyperparamètres
   - Test de différents modèles

4. **Analyse et Recommandations**
   - Analyse des paramètres du meilleur modèle
   - Identification des leviers d'action
   - Recommandations pour améliorer le taux de conversion

## Métriques d'Évaluation

Le modèle est évalué en utilisant le F1-score, qui est particulièrement adapté pour les problèmes de classification déséquilibrés.

## Prérequis

- Python 3.x
- Bibliothèques Python requises :
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - scikit-learn

## Installation

1. Clonez le repository
2. Installez les dépendances :
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Utilisation

1. Ouvrez le notebook `main.ipynb`
2. Suivez les étapes d'analyse et de modélisation
3. Générez les prédictions pour le fichier de test
4. Soumettez les prédictions pour évaluation

## Livrables Attendus

- Visualisations pertinentes pour l'EDA
- Au moins un modèle prédictif avec évaluation des performances
- Au moins une soumission au leaderboard
- Analyse des paramètres du meilleur modèle
- Recommandations pour améliorer le taux de conversion

## Auteur

Gilles Akakpo 