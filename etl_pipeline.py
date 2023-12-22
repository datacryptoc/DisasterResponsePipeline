import pandas as pd
import numpy as np
import sqlalchemy
from sqlalchemy import create_engine

def etl_pipeline():
    # Carga dataset mensajes
    messages = pd.read_csv("data/messages.csv")

    # Cargar dataset categorias
    categories = pd.read_csv("data/categories.csv")
    ids = categories['id']
    categories = categories['categories'].str.split(';', expand=True)

    # Obtener los nombres de las categorías y asignarlos como nombres de columnas
    category_colnames = categories.iloc[0].apply(lambda x: x.split('-')[0])
    categories.columns = category_colnames

    # Convertir los valores de las categorías a números
    for column in categories:
        categories[column] = categories[column].str.split('-').str.get(1).astype(int)

    categories = pd.concat([ids, categories], axis=1)

    # Join Datasets
    df = pd.merge(messages, categories, on="id")

    # Convierto a minúsculas
    df["message"] = df["message"].str.lower()

    # Eliminar duplicados.
    df = df.drop_duplicates()

    # Generar base de datos
    engine = create_engine('sqlite:///data/DisasterMessages.db')
    df.to_sql('data/DisasterMessages', engine, index=False)

    return df