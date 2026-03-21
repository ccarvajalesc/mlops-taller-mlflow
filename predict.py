import pandas as pd
import joblib
import mlflow
import os
# Columnas esperadas
EXPECTED_NUM_COLS = [
    "culmen_length_mm",
    "culmen_depth_mm",
    "flipper_length_mm",
    "body_mass_g"
]

EXPECTED_CAT_COLS = ["island", "sex"]

ALL_EXPECTED_COLS = EXPECTED_NUM_COLS + EXPECTED_CAT_COLS

os.environ["MLFLOW_TRACKING_URI"] = "http://mlflow:5000"

os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://minio:9000"
os.environ["AWS_ACCESS_KEY_ID"] = "admin"
os.environ["AWS_SECRET_ACCESS_KEY"] = "supersecret"
os.environ["PASSWORD"] = "valentasecret"


def load_model_from_mlflow(model_name="penguins_decision_tree_model", stage="Production"):
    """
    Carga el modelo desde MLflow Model Registry usando el stage (ej: Production).
    
    Args:
        model_name (str): Nombre del modelo registrado en MLflow.
        stage (str): Stage del modelo (ej: 'Production', 'Staging').
    
    Returns:
        loaded_model: Modelo cargado como pyfunc.
    """
    model_uri = f"models:/{model_name}/{stage}"
    loaded_model = mlflow.pyfunc.load_model(model_uri=model_uri)
    return loaded_model


def load_encoder(path="encoders/ohe_encoder.joblib"):
    """
    Carga el OneHotEncoder desde archivo joblib.
    
    Args:
        path (str): Ruta al archivo del encoder.
    
    Returns:
        ohe: OneHotEncoder entrenado.
    """
    return joblib.load(path)


def preprocess_input(df_new, ohe):
    """
    Preprocesa los datos de entrada para que coincidan con el entrenamiento.
    
    Args:
        df_new (pd.DataFrame): Nuevos datos crudos.
        ohe: OneHotEncoder cargado.
    
    Returns:
        pd.DataFrame: DataFrame listo para predicción.
    """
    df_new = df_new.copy()

    # Rellenar valores faltantes
    df_new[EXPECTED_NUM_COLS] = df_new[EXPECTED_NUM_COLS].fillna(df_new[EXPECTED_NUM_COLS].median())
    df_new["sex"] = df_new["sex"].fillna("Unknown")

    # Codificar categóricas
    X_cat = ohe.transform(df_new[EXPECTED_CAT_COLS])
    cat_feature_names = ohe.get_feature_names_out(EXPECTED_CAT_COLS)
    X_cat_df = pd.DataFrame(X_cat, columns=cat_feature_names, index=df_new.index)

    # Concatenar numéricas y categóricas
    X_final = pd.concat([df_new[EXPECTED_NUM_COLS], X_cat_df], axis=1)

    return X_final


def predict_new_data(df_new, model, ohe):
    """
    Realiza predicciones sobre nuevos datos usando el modelo cargado desde MLflow.
    
    Args:
        df_new (pd.DataFrame): Nuevos datos crudos.
        model: Modelo cargado desde MLflow.
        ohe: OneHotEncoder cargado.
    
    Returns:
        np.ndarray: Predicciones del modelo.
    """
    X_final = preprocess_input(df_new, ohe)
    return model.predict(X_final)
