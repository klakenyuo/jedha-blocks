# Détecteur de Spam SMS pour AT&T

Ce projet est un détecteur de spam SMS utilisant le deep learning pour classer automatiquement les messages en spam ou ham (messages légitimes). Il a été développé pour AT&T dans le cadre d'une solution automatisée de détection de spam.

## Contexte du Projet

AT&T, l'un des plus grands opérateurs de télécommunications au monde, fait face à un défi constant : la protection de ses utilisateurs contre les messages spam. Alors que la détection manuelle des spams a été utilisée pendant des années, l'entreprise cherche maintenant à automatiser ce processus pour une meilleure efficacité et une protection en temps réel.

## Objectifs

- Développer un modèle de deep learning capable de détecter automatiquement les messages spam
- Atteindre une précision élevée dans la classification des messages
- Créer une solution scalable et maintenable
- Fournir des visualisations claires des performances du modèle

## Structure du Projet

```
spam_detector/
├── main.py              # Code principal du détecteur de spam
├── requirements.txt     # Dépendances du projet
├── spam.csv            # Dataset d'entraînement
├── spam_detector_model.h5  # Modèle entraîné
├── training_history.png    # Visualisations des performances
└── README.md           # Documentation du projet
```

## Prérequis

- Python 3.8 ou supérieur
- pip (gestionnaire de paquets Python)
- 4GB de RAM minimum
- GPU recommandé pour l'entraînement (mais non obligatoire)

## Installation

1. Clonez le repository :
```bash
git clone [URL_DU_REPO]
cd spam_detector
```

2. Créez un environnement virtuel (recommandé) :
```bash
python -m venv venv
source venv/bin/activate  # Sur Unix/macOS
# ou
venv\Scripts\activate  # Sur Windows
```

3. Installez les dépendances :
```bash
pip install -r requirements.txt
```

## Structure des Données

Le dataset (`spam.csv`) contient :
- `label` : Classification du message (spam ou ham)
- `message` : Contenu du message SMS

Statistiques du dataset :
- Nombre total de messages : 5573
- Distribution des classes : ~87% ham, ~13% spam

## Architecture du Modèle

Le modèle utilise une architecture LSTM (Long Short-Term Memory) avec les composants suivants :
- Couche d'embedding (32 dimensions)
- Deux couches LSTM (64 et 32 unités)
- Couches de dropout (0.2) pour la régularisation
- Couches denses (16 et 1 unité)

### Hyperparamètres
- Taille maximale du vocabulaire : 10000 mots
- Longueur maximale des séquences : 100 tokens
- Taille du batch : 32
- Nombre d'époques : 10 (avec early stopping)
- Validation split : 20%

## Fonctionnalités

- Prétraitement automatique des textes
- Gestion de multiples encodages de caractères
- Tokenization et padding des séquences
- Entraînement avec early stopping
- Visualisation des métriques d'entraînement
- Sauvegarde automatique du modèle
- Évaluation des performances sur l'ensemble de test

## Utilisation

Pour exécuter le détecteur de spam :

```bash
python main.py
```

Le script va :
1. Charger et prétraiter les données
2. Entraîner le modèle
3. Évaluer les performances
4. Générer des visualisations
5. Sauvegarder le modèle entraîné

## Sorties

Le script génère deux fichiers :
- `spam_detector_model.h5` : Le modèle entraîné (peut être utilisé pour des prédictions futures)
- `training_history.png` : Graphiques montrant l'évolution de l'accuracy et de la loss pendant l'entraînement

## Métriques de Performance

Le modèle est évalué sur :
- Accuracy (précision globale)
- Loss (perte)
- Validation accuracy
- Validation loss

Les performances attendues :
- Accuracy sur l'ensemble de test : > 95%
- F1-score pour la classe spam : > 0.85

## Limitations et Améliorations Futures

- Le modèle est optimisé pour les messages en anglais
- La détection de nouveaux types de spam pourrait nécessiter un réentraînement
- Possibilité d'améliorer les performances avec :
  - Augmentation de la taille du dataset
  - Utilisation de techniques de transfer learning
  - Optimisation des hyperparamètres
  - Ajout de features supplémentaires (longueur du message, présence de liens, etc.)

## Auteurs

- [Votre Nom]

## Licence

Ce projet est sous licence MIT. 