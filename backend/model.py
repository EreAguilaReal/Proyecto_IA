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


# Groups preserve dependencies between input variables during replacement.
EXPLANATION_GROUPS = {
    "adeudo": ("Monto y pagos parciales", ("monto_pendiente", "pagos_parciales")),
    "dias": ("Días de vencimiento", ("dias_vencimiento",)),
    "incumplimientos": ("Incumplimientos anteriores", ("incumplimientos_anteriores",)),
    "contacto": ("Intentos y respuesta de contacto", ("intentos_contacto", "respuesta_contacto")),
}


def training_references(x_train: pd.DataFrame) -> dict:
    """Select observed training values near medians, or the most common contact pair."""
    references = {}
    for key, (_, columns) in EXPLANATION_GROUPS.items():
        values = x_train[list(columns)]
        if key == "contacto":
            # Stable tie-breaking makes retraining reproducible.
            counts = values.value_counts().sort_index()
            selected = counts.idxmax()
        else:
            medians = values.median()
            scales = (values.max() - values.min()).replace(0, 1)
            distances = ((values - medians).abs() / scales).sum(axis=1)
            selected = tuple(values.loc[distances.idxmin(), list(columns)])
        references[key] = {column: float(value) for column, value in zip(columns, selected, strict=True)}
    return references
