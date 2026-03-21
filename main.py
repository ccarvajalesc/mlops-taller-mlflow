"""
Penguin Prediction API
"""

from fastapi import FastAPI, Query
import pandas as pd
from typing import List, Annotated
from enum import Enum
import logging

from predict import predict_new_data, load_model_from_mlflow, load_encoder
#from model_registry import get_model   # ⭐ nuevo

# ------------------------------------------------------------------------------
# Inicialización
# ------------------------------------------------------------------------------

app = FastAPI(
    title="Taller 1 MLOPS: Penguin Prediction API",
    description="Servicio para clasificar pingüinos",
    version="1.0.0",
)

logging.basicConfig(
    filename="logs/app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# Enumeraciones
# ------------------------------------------------------------------------------

class islas_class(str, Enum):
    Torgersen = "Torgersen"
    Dream = "Dream"
    Biscoe = "Biscoe"


class sex_class(str, Enum):
    Male = "Male"
    Female = "Female"


class model_class(str, Enum):
    TREE = "TREE"
    KNN = "KNN"
    SVM = "SVM"


# ⭐ mapa modelo → archivo
model_map = {
    "TREE": "penguins_decision_tree_model",
    "KNN": "penguins_knn_model",
    "SVM": "penguins_svm_model"
}

# ------------------------------------------------------------------------------
# Endpoint
# ------------------------------------------------------------------------------

@app.post("/predict")
async def predict_endpoint(
    models: Annotated[List[model_class], Query(...)],
    culmen_length_mm: float = Query(39),
    culmen_depth_mm: float = Query(18.7),
    flipper_length_mm: float = Query(180),
    body_mass_g: float = Query(3700),
    island: islas_class = Query(islas_class.Torgersen),
    sex: sex_class = Query(sex_class.Male),
):

    # construir input
    df = pd.DataFrame([{
        "culmen_length_mm": culmen_length_mm,
        "culmen_depth_mm": culmen_depth_mm,
        "flipper_length_mm": flipper_length_mm,
        "body_mass_g": body_mass_g,
        "island": island,
        "sex": sex
    }])

    logger.info(f"input recibido: {df.to_json()}")

    response = {}

    # ⭐ aquí ocurre el hot reload
    for m in models:
        filename = model_map[m.value]

        bundle = load_model_from_mlflow(model_name=filename, stage="Production")  # ⭐ revisa timestamp y recarga si cambió

        prediction = predict_new_data(
            df,
            bundle,
            load_encoder()  # ⭐ también recarga el encoder si cambió
        )

        response[m.value] = prediction.tolist()

    logger.info(f"Response enviado: {response}")

    return response