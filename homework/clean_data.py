"""Taller evaluable presencial"""

#
# Este codigo implementa algortimo 'fingerprint' para colision de textos, el 
# cual es utilizado para unificar cadenas de texto que representan la misma
# entidad.
# 
# Referencia:
# https://openrefine.org/docs/technical-reference/clustering-in-depth
# 
import nltk 
import pandas as pd 


def load_data(input_file):
    """Lea el archivo usando pandas y devuelva un DataFrame"""

    #
    # Esta parte es igual al taller anterior
    #
    df = pd.read_csv(input_file)
    return df


def create_key(df, n):
    """Cree una nueva columna en el DataFrame que contenga el key de la
    columna 'text'"""

    df = df.copy()
    df["key"] = df["raw_text"]
    df["key"] = df["key"].str.strip()
    df["key"] = df["key"].str.lower()
    df["key"] = df["key"].str.replace("-", "")
    df["key"] = df["key"].str.translate(
        str.maketrans("", "", "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")
    )
    df["key"] = df["key"].str.split()

    
    # Transforme cada palabra con un stemmer de Porter
    stemmer = nltk.stem.PorterStemmer()
    df["key"] = df["key"].apply(lambda x: [stemmer.stem(word) for word in x])
    
    df["key"] = df["key"].apply(lambda x: sorted(set(x)))
    
    df["key"] = df["key"].str.join(" ")
    return df


def generate_cleaned_text(df):
    """Crea la columna 'cleaned_text' en el DataFrame"""

    keys = df.copy()
    
    # Ordene el dataframe por 'key' y 'text'
    keys = keys.sort_values(by=["key", "raw_text"], ascending=[True, True])
    
    # Seleccione la primera fila de cada grupo de 'key' 
    keys = df.drop_duplicates(subset="key", keep="first")
    
    # Cree un diccionario con 'key' como clave y 'raw_text' como valor
    key_dict = dict(zip(keys["key"], keys["raw_text"]))
    
    # Cree la columna 'cleaned' usando el diccionario
    df["cleaned_text"] = df["key"].map(key_dict)
    
    return df


def save_data(df, output_file):
    """Guarda el DataFrame en un archivo"""
    #
    # Este código es identico al anteior
    #
    df = df.copy()
    df = df[["raw_text", "cleaned_text"]]
    df.to_csv(output_file, index=False)


def main(input_file, output_file, n=2):
    """Ejecuta la limpieza de datos"""
    #
    # Este código es identico al anteior
    #
    df = load_data(input_file)
    df = create_key(df, n)
    df = generate_cleaned_text(df)
    df.to_csv("files/test.csv", index=False)
    save_data(df, output_file)


if __name__ == "__main__":
    main(
        input_file="files/input.txt",
        output_file="files/output.txt",
    )

