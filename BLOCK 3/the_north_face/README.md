# The North Face - Système de Recommandation de Produits

## Description du Projet 🎯

Ce projet vise à améliorer l'expérience d'achat en ligne sur le site web de The North Face en implémentant des solutions d'intelligence artificielle pour :
1. Créer un système de recommandation de produits
2. Optimiser la structure du catalogue de produits grâce à l'extraction de topics

## Fonctionnalités Principales 🚀

### 1. Clustering des Produits
- Utilisation de DBSCAN pour regrouper les produits similaires
- Visualisation des clusters via des wordclouds
- Paramètres optimisés pour obtenir 10-20 clusters

### 2. Système de Recommandation
- Recommandation de 5 produits similaires basée sur le clustering
- Interface utilisateur interactive
- Prise en compte des descriptions de produits pour la similarité

### 3. Topic Modeling
- Extraction de 15 topics latents via TruncatedSVD
- Visualisation des topics avec des wordclouds
- Attribution d'un topic principal à chaque produit

## Prérequis Techniques 🛠️

### Installation des Dépendances

#### Option 1 : Installation via requirements.txt (recommandée)
```bash
pip install -r requirements.txt
```

#### Option 2 : Installation manuelle
```bash
pip install pandas numpy scikit-learn spacy wordcloud matplotlib
python -m spacy download en_core_web_sm
```

### Bibliothèques Utilisées
- pandas : Manipulation des données
- numpy : Calculs numériques
- scikit-learn : Machine Learning (DBSCAN, TruncatedSVD, TF-IDF)
- spacy : Traitement du langage naturel
- wordcloud : Visualisation des mots-clés
- matplotlib : Visualisation des graphiques

## Structure du Projet 📁

```
the_north_face/
├── data.csv           # Données des produits
├── main.py           # Code principal
├── description.ipynb # Description du projet
├── requirements.txt  # Dépendances du projet
└── README.md        # Documentation
```

## Utilisation du Programme 🎮

1. Lancez le programme :
```bash
python main.py
```

2. Menu interactif disponible :
   - Option 1 : Afficher les wordclouds des clusters
   - Option 2 : Trouver des produits similaires
   - Option 3 : Afficher les wordclouds des topics
   - Option 4 : Quitter

## Traitement des Données 🔄

### Préprocessing
1. Nettoyage des descriptions :
   - Suppression des balises HTML
   - Conversion en minuscules
   - Lemmatisation avec spaCy

2. Vectorisation :
   - Création de la matrice TF-IDF
   - Limitation à 1000 features

### Clustering
- Algorithme : DBSCAN
- Métrique : Cosine
- Paramètres :
  - eps = 0.3
  - min_samples = 5

### Topic Modeling
- Algorithme : TruncatedSVD
- Nombre de topics : 15
- Visualisation via wordclouds

## Résultats et Visualisations 📊

### Clusters
- Chaque cluster représente un groupe de produits similaires
- Visualisation via wordclouds montrant les mots-clés dominants

### Recommandations
- Pour chaque produit, 5 recommandations basées sur la similarité
- Prise en compte du cluster d'appartenance

### Topics
- 15 topics extraits des descriptions
- Visualisation des mots-clés dominants par topic
- Attribution d'un topic principal à chaque produit

## Améliorations Possibles 🔄

1. Optimisation des paramètres :
   - Ajustement des paramètres DBSCAN
   - Variation du nombre de topics
   - Modification du nombre de features TF-IDF

2. Ajout de fonctionnalités :
   - Interface graphique
   - Export des résultats
   - Métriques de performance

3. Amélioration du preprocessing :
   - Gestion des stop words personnalisés
   - Nettoyage plus poussé des descriptions
   - Ajout de features supplémentaires
 