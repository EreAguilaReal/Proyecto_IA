from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ARTIFACT_PATH = Path(__file__).resolve().parent / "artifacts" / "modelo_cobranza_rf.skops"
MODEL_NAME = "Random Forest"
DATA_SOURCE = "synthetic"
NUMERIC_FEATURES = [
    "monto_pendiente",
    "dias_vencimiento",
    "incumplimientos_anteriores",
    "pagos_parciales",
    "porcentaje_abonado",
    "intentos_contacto",
]
FEATURES = [*NUMERIC_FEATURES, "respuesta_contacto"]


def prepare_features(frame: pd.DataFrame) -> pd.DataFrame:
    features = frame.copy()
    monto_original = features["monto_pendiente"] + features["pagos_parciales"]
    features["porcentaje_abonado"] = features["pagos_parciales"] / monto_original
    return features[FEATURES]


def make_pipeline(classifier) -> Pipeline:
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(drop="first"), ["respuesta_contacto"]),
    ])
    return Pipeline([("prep", preprocessor), ("clf", classifier)])
