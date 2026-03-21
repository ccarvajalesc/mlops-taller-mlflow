import os
import time
import pandas as pd
from sqlalchemy import create_engine, text, inspect, MetaData
import mysql.connector

from sklearn.preprocessing import OneHotEncoder
#print("Dir: ",os.listdir())


def preprocess_data(df):
    """Preprocesa los datos del conjunto de datos de pingüinos usando OneHotEncoder para variables categóricas.
    
    Operaciones:
    - Rellena valores faltantes en columnas numéricas con la mediana
    - Rellena valores faltantes en columnas categóricas con "Unknown"
    - Codifica las variables categóricas con OneHotEncoder
    - Devuelve el DataFrame completo ya preprocesado, además de X, y y los encoders
    
    Args:
        df (pd.DataFrame): DataFrame con los datos crudos de pingüinos.
        
    Returns:
        tuple: 
            - df_preprocessed (pd.DataFrame): DataFrame completo preprocesado
            - X (pd.DataFrame): Características preprocesadas
            - y (pd.Series): Variable objetivo
            - encoders (dict): Diccionario con los OneHotEncoders
    """
    df = df.copy()

    num_cols = [
        "culmen_length_mm",
        "culmen_depth_mm",
        "flipper_length_mm",
        "body_mass_g"
    ]

    cat_cols = ["island", "sex"]
    target = "species"

    # Rellenar valores faltantes
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    df["sex"] = (
        df["sex"]
        .replace([".", "", "NA", "N/A"], "Unknown")
        .fillna("Unknown")
    )
    df["sex"] = df["sex"].where(
        df["sex"].isin(["MALE", "FEMALE"]),
        "Unknown"
    )
    # Separar target
    y = df[target]
    X = df.drop(columns=target)

    # OneHotEncoder para variables categóricas
    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    X_cat = ohe.fit_transform(X[cat_cols])
    cat_feature_names = ohe.get_feature_names_out(cat_cols)
    X_cat_df = pd.DataFrame(X_cat, columns=cat_feature_names, index=X.index)

    # Concatenar numéricas y categóricas
    X_final = pd.concat([X[num_cols], X_cat_df], axis=1)

    # Crear df preprocesado (incluye target)
    df_preprocessed = pd.concat([X_final, y], axis=1)

    encoders = {"onehot": ohe}

    return df_preprocessed, X_final, y, encoders



def get_engine():
    """Create SQLAlchemy engine from env vars."""
    user = os.getenv("MYSQL_USER", "mlops_user")
    password = os.getenv("MYSQL_PASSWORD", "mlops_pass")
    host = os.getenv("MYSQL_HOST", "mysql_db")
    port = os.getenv("MYSQL_PORT", "3306")
    db = os.getenv("MYSQL_DB", "mlops_db")

    url = f"mysql+pymysql://{user}:{password}@{host}:{port}/{db}"
    return create_engine(url)

def clear_database():
    """Limpia las tablas de la base de datos usando SQLAlchemy"""

    engine = get_engine()
    wait_for_db(engine)

    inspector = inspect(engine)

    with engine.begin() as conn:

        if "penguins_raw" in inspector.get_table_names():
            conn.execute(text("DROP TABLE penguins_raw"))

        if "penguins_processed" in inspector.get_table_names():
            conn.execute(text("DROP TABLE penguins_processed"))

    print("✅ Tables dropped")

def wait_for_db(engine, retries=10, sleep=3):
    """Wait until DB is ready (important in Docker)."""
    for i in range(retries):
        try:
            with engine.connect():
                print("✅ DB ready")
                return
        except Exception as e:
            print(f"⏳ waiting for DB... ({i+1}/{retries})")
            time.sleep(sleep)
    raise RuntimeError("Database not reachable")


def preprocess_data_for_training():
    """Preprocesa los datos"""
    
    engine = get_engine()
    wait_for_db(engine)

    # Leer usando SQLAlchemy (sin conexión manual)
    df = pd.read_sql_table("penguins_raw", con=engine)

    #df = df.dropna()
    
    df_processed, _, _, encoders = preprocess_data(df)

    df_processed.to_sql(
        "penguins_processed",
        con=engine,
        if_exists="replace",
        index=False,
        method="multi"
    )

    import joblib
    joblib.dump(encoders["onehot"], "encoders/ohe_encoder.joblib")

    print(f"✅ Processed {len(df_processed)} rows")

def load_penguins(csv_path: str):
    engine = get_engine()
    wait_for_db(engine)

    print("📥 Reading CSV...")
    df = pd.read_csv(csv_path)

    # Asegurarse de que las columnas coincidan con el pipeline y la base de datos
    expected_cols = [
        "culmen_length_mm",
        "culmen_depth_mm",
        "flipper_length_mm",
        "body_mass_g",
        "island",
        "sex",
        "species"
    ]
    if not all(col in df.columns for col in expected_cols):
        raise ValueError(f"El CSV debe contener las columnas: {expected_cols}")

    print("🗄️ Writing to MySQL (penguins_raw)...")
    df[expected_cols].to_sql(
        "penguins_raw",
        con=engine,
        if_exists="append",  # importante para pipeline real
        index=False,
        chunksize=1000,
    )

    print(f"✅ Loaded {len(df)} rows")
if __name__ == "__main__":
    clear_database()
    load_penguins("datasets/penguins_size.csv")
    preprocess_data_for_training()