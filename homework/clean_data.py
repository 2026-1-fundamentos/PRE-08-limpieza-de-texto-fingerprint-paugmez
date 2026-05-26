"""taller evaluable presencial"""

#
# este coddigo implementa algoritmo 'fingerprint' para colision de textos, el
# cual es utilizado para unificar cadenas de texto que representan la misma
# entidad
#
# Referencia:
# https://openrefine.org/docs/technical-reference/clustering-in-depth
#
import nltk #type: ignore
import pandas as pd #type: ignore


def load_data(input_file):
    """lea el archivo usando pandas y devuelva un Dataframe"""
    
    df = pd.read_csv(input_file)
    return df


def create_normalized_key(df):
    """Cree una nueva columna en el DataFrame que contenga 
    el key de la columna 'raw_test'"""
    
    df = df.copy()
    
    #copie la columna 'text' a la columna 'key'
    df["key"] = df["raw_test"]
    
    #remueva los espacios en blanco al inicio y al final de la cadena
    df["key"] = df["key"].str.strip()
    
    #convierta el texto a minúsculas
    df["key"] = df["key"].str.lower()
    
    #TRansforme palabras que pueden (o no) contener guiones por su
    # version sin guion (este paso es redundante por la limea siguiente.
    # pero es claro anotar la existencia de palabras con y sin '-'.
    df["key"] = df["key"].str.replace("-", "")
    
    #remueva puntuacion y caracteres de control
    df["key"] = df["key"].str.translate(
        str.maketrans("", "", "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")
    )
    
    #convierta el texto a una lista de tokens 
    df["key"] = df["key"].str.split()
    
    #transforme cada palabra en un stemmer de porter
    stemmer = nltk.PorterStemmer()
    df["key"] = df["key"].apply(lambda x: [stemmer.stem(word) for word in x])
    
    #ordene la lista de tokens y remueva duplicados
    df["key"] = df["key"].apply(lambda x: sorted(set(x)))
    
    #convierta la lista de tokens a una cadena de texto separada por espacios
    df["key"] = df["key"].str.join(" ")
    
    return df


def generate_cleaned_test(df):
    """crea la columna 'cleaned_test' en el dataframe"""
    
    keys = df.copy()
    
    #ordene el dataframe por 'Key' y 'text'
    keys= keys.sort_values(by=["key", "raw_test"], ascending=[True, True])
    
    #seleccione la primera fila de cada grupo de 'key'
    keys = df.drop_duplicates(subset="key", keep="first")
    
    #cree un diccionario con 'key' como clave y 'test' como valor
    key_dict = dict(zip(keys["key"], keys["raw_test"]))
    
    #cree la columna 'cleaned' usando el diccionario
    df["cleaned_test"] = df["key"].map(key_dict)
    
    return df


def save_data(df, output_file): 
    """guarda el dataframe en un archivo"""
    
    df= df.copy()
    df=df[["raw_test", "cleaned_test"]]
    df.to_csv(output_file, index=False) 
    
    
def main(input_file, output_file):
    """Ejecuta la limpieza de datos"""
    
    df = load_data(input_file)
    df = create_normalized_key(df)
    df = generate_cleaned_test(df)
    df.to_csv("files/test.csv", index=False)
    save_data(df, output_file)
    
if __name__ == "__main__":
    main(
        input_file="files/input.txt",
        output_file="files/output.txt"
    )