import pandas as pd
import numpy as np
import sqlalchemy
from sqlalchemy import create_engine
import re
import pickle
import subprocess
import sys

import nltk

# nltk.download("punkt")
# nltk.download("stopwords")
# nltk.download("words")
# nltk.download('wordnet')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def load_data():
    """
    Función para cargar datos de sqllite.
    :return list X: Lista de frases o features.
    :return DataFrme y: Datafrme con targets.
    """
    engine = create_engine('sqlite:///data/DisasterMessages.db')
    df = pd.read_sql_table('data/DisasterMessages', engine)

    out_cols = ["id", "original", "genre"]
    features_cols = ["message"]
    target_cols = [x for x in df.columns if x not in [*out_cols, *features_cols]]
    X = df["message"]
    y = df[target_cols]
    return X, y

def tokenize(sentence):
    """
    Función para lipiar y tokenizar una lista de frases.
    :params list X: Lista de frases a tokenizar.
    :return list sentences: Lista de frases tokenizadas.
    """
    # Normalización a minúsculas
    tokens = sentence.lower()
    # Eliminación carácteres no alfanuméricos.
    tokens = re.sub(r'[^a-zA-Z0-9 ]', '', tokens)
    # Tokenización
    tokens = word_tokenize(tokens)
    # Eliminación stopwords
    sw = stopwords.words("english")
    tokens = [w for w in tokens if w not in sw]
    # Steaming
    tokens = [PorterStemmer().stem(w) for w in tokens]
    # Lematización
    tokens = [WordNetLemmatizer().lemmatize(w) for w in tokens]
    return tokens

def display_report(y_test, y_pred):
    """
    Función para representar reporte de métricas de clasificación.
    :params DataFrame y_test:
    :params DataFrame y_pred:
    :return str report: Reporte de la predicción.
    """
    y_test = y_test.to_numpy()
    for i in range(y_test.shape[1]):
        print(f"Clase {i}:")
        print(classification_report(y_test.iloc[:, i], y_pred[:, i]))

def main():
    X, y = load_data()

    #División train, test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # Creación del Pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', RandomForestClassifier())
    ])
        
    # Entrenamiento del modelo
    pipeline.fit(X_train, y_train)

    # Guardado modelo
    with open('models/modelo_entrenado_1.pkl', 'wb') as file:
        pickle.dump(pipeline, file)

    # Predicción y métricas
    y_pred = pipeline.predict(X_test)
    display_report(y_test, y_pred)