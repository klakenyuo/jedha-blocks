import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
from wordcloud import WordCloud
from sklearn.metrics import confusion_matrix, classification_report
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import warnings
warnings.filterwarnings('ignore')

# Télécharger les ressources NLTK nécessaires
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# 1. Chargement et préparation des données
def load_data():
    # Essayer différents encodages
    encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
    
    for encoding in encodings:
        try:
            # Charger le dataset avec l'encodage actuel
            df = pd.read_csv('spam.csv', encoding=encoding)
            print(f"Fichier chargé avec succès avec l'encodage {encoding}")
            break
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"Erreur avec l'encodage {encoding}: {str(e)}")
            continue
    else:
        raise ValueError("Impossible de charger le fichier avec aucun des encodages supportés")
    
    # Renommer les colonnes
    df.columns = ['label', 'message']
    
    # Encoder les labels
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['label'])
    
    return df

def analyze_data(df):
    """Analyse exploratoire des données"""
    print("\n=== Analyse Exploratoire des Données ===")
    
    # 1. Statistiques de base
    print("\nStatistiques de base :")
    print(f"Nombre total de messages : {len(df)}")
    print(f"Distribution des classes :\n{df['label'].value_counts(normalize=True)}")
    
    # 2. Longueur des messages
    df['message_length'] = df['message'].str.len()
    print("\nStatistiques sur la longueur des messages :")
    print(df['message_length'].describe())
    
    # 3. Visualisations
    plt.figure(figsize=(15, 10))
    
    # Distribution des classes
    plt.subplot(2, 2, 1)
    sns.countplot(data=df, x='label')
    plt.title('Distribution des Classes')
    plt.xlabel('Label (0: Ham, 1: Spam)')
    plt.ylabel('Nombre de messages')
    
    # Distribution de la longueur des messages
    plt.subplot(2, 2, 2)
    sns.boxplot(data=df, x='label', y='message_length')
    plt.title('Longueur des Messages par Classe')
    plt.xlabel('Label (0: Ham, 1: Spam)')
    plt.ylabel('Longueur (caractères)')
    
    # Word Cloud pour les messages ham
    plt.subplot(2, 2, 3)
    ham_words = ' '.join(df[df['label'] == 0]['message'])
    wordcloud_ham = WordCloud(width=800, height=400, background_color='white').generate(ham_words)
    plt.imshow(wordcloud_ham, interpolation='bilinear')
    plt.axis('off')
    plt.title('Nuage de Mots - Messages Ham')
    
    # Word Cloud pour les messages spam
    plt.subplot(2, 2, 4)
    spam_words = ' '.join(df[df['label'] == 1]['message'])
    wordcloud_spam = WordCloud(width=800, height=400, background_color='white').generate(spam_words)
    plt.imshow(wordcloud_spam, interpolation='bilinear')
    plt.axis('off')
    plt.title('Nuage de Mots - Messages Spam')
    
    plt.tight_layout()
    plt.savefig('data_analysis.png')
    
    # 4. Analyse des mots les plus fréquents
    def get_most_common_words(texts, n=10):
        words = ' '.join(texts).lower()
        words = re.findall(r'\w+', words)
        stop_words = set(stopwords.words('english'))
        words = [w for w in words if w not in stop_words]
        return Counter(words).most_common(n)
    
    print("\nMots les plus fréquents dans les messages Ham :")
    print(get_most_common_words(df[df['label'] == 0]['message']))
    
    print("\nMots les plus fréquents dans les messages Spam :")
    print(get_most_common_words(df[df['label'] == 1]['message']))

# 2. Prétraitement du texte
def preprocess_text(texts, max_words=10000, max_len=100):
    # Créer le tokenizer
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(texts)
    
    # Convertir les textes en séquences
    sequences = tokenizer.texts_to_sequences(texts)
    
    # Padding des séquences
    padded_sequences = pad_sequences(sequences, maxlen=max_len)
    
    return padded_sequences, tokenizer

# 3. Création du modèle
def create_model(max_words, max_len):
    model = Sequential([
        Embedding(max_words, 32, input_length=max_len),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam',
                 loss='binary_crossentropy',
                 metrics=['accuracy'])
    
    return model

def plot_training_history(history):
    """Visualisation de l'historique d'entraînement"""
    plt.figure(figsize=(15, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')

def plot_confusion_matrix(y_true, y_pred):
    """Visualisation de la matrice de confusion"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Matrice de Confusion')
    plt.xlabel('Prédictions')
    plt.ylabel('Valeurs Réelles')
    plt.savefig('confusion_matrix.png')

# 4. Fonction principale
def main():
    # Charger les données
    df = load_data()
    
    # Analyse exploratoire des données
    analyze_data(df)
    
    # Diviser en train et test
    X_train, X_test, y_train, y_test = train_test_split(
        df['message'], df['label'], test_size=0.2, random_state=42
    )
    
    # Prétraiter les textes
    max_words = 10000
    max_len = 100
    X_train_padded, tokenizer = preprocess_text(X_train, max_words, max_len)
    X_test_padded = tokenizer.texts_to_sequences(X_test)
    X_test_padded = pad_sequences(X_test_padded, maxlen=max_len)
    
    # Créer et entraîner le modèle
    model = create_model(max_words, max_len)
    
    # Early stopping pour éviter le surapprentissage
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )
    
    # Entraînement
    history = model.fit(
        X_train_padded, y_train,
        epochs=10,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping]
    )
    
    # Évaluer le modèle
    test_loss, test_accuracy = model.evaluate(X_test_padded, y_test)
    print(f"\nTest accuracy: {test_accuracy:.4f}")
    
    # Prédictions
    y_pred = (model.predict(X_test_padded) > 0.5).astype(int)
    
    # Visualisations
    plot_training_history(history)
    plot_confusion_matrix(y_test, y_pred)
    
    # Rapport de classification
    print("\nRapport de classification :")
    print(classification_report(y_test, y_pred))
    
    # Sauvegarder le modèle
    model.save('spam_detector_model.h5')

if __name__ == "__main__":
    main()
