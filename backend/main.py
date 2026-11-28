from functools import lru_cache
import logging
from pathlib import Path
from typing import Annotated, Literal, NamedTuple

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from skops.io import load

from .data import sample_records
from .model import ARTIFACT_PATH, DATA_SOURCE, FEATURES, MODEL_NAME, prepare_features


logger = logging.getLogger(__name__)
app = FastAPI(title="Priorización de cobranza")
FRONTEND = Path(__file__).resolve().parent.parent / "frontend"
TRUSTED_MODEL_TYPES = ["sklearn.compose._column_transformer._RemainderColsList"]

Identifier = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=128)]
Money = Annotated[float, Field(ge=0, le=1e30, allow_inf_nan=False)]
Count = Annotated[int, Field(ge=0, le=2**53 - 1)]
Probability = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]


class DebtCase(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    id_deudor: Identifier
    monto_pendiente: Money
    dias_vencimiento: Count
    incumplimientos_anteriores: Count
    pagos_parciales: Money
    intentos_contacto: Count
    respuesta_contacto: Annotated[int, Field(ge=0, le=2)]
    situacion_adeudo: Annotated[int, Field(ge=0, le=1)]

    @model_validator(mode="after")
    def validate_case(self):
        if self.intentos_contacto == 0 and self.respuesta_contacto != 0:
            raise ValueError("Con cero intentos de contacto, la respuesta debe ser 0.")
        if self.situacion_adeudo == 1 and self.monto_pendiente + self.pagos_parciales == 0:
            raise ValueError("Un adeudo disponible debe tener monto pendiente o pagos parciales.")
        return self


class CaseInput(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    cases: Annotated[list[DebtCase], Field(min_length=1)]

    @model_validator(mode="after")
    def validate_unique_ids(self):
        seen = set()
        for case in self.cases:
            if case.id_deudor in seen:
                raise ValueError(f"El ID de deudor está duplicado: {case.id_deudor}.")
            seen.add(case.id_deudor)
        return self


class ScoredCase(DebtCase):
    probability: Probability
    rank_model: int
    rank_amount: int
    rank_days: int


class OutOfRange(BaseModel):
    id_deudor: str
    fields: list[str]


class CaseAnalysis(BaseModel):
    cases: list[ScoredCase]
    excluded: list[DebtCase]
    warnings: list[OutOfRange]


class Scores(BaseModel):
    f1: Probability
    roc_auc: Probability


class ModelMetrics(BaseModel):
    naive_bayes: Scores
    random_forest: Scores


class ModelStatus(BaseModel):
    ready: bool
    model: Literal["Random Forest"] = MODEL_NAME
    data_source: Literal["synthetic"] = DATA_SOURCE
    metrics: ModelMetrics | None = None


class ModelArtifact(NamedTuple):
    pipeline: Pipeline
    metrics: ModelMetrics
    feature_bounds: dict[str, tuple[float, float]]


@app.exception_handler(RequestValidationError)
async def validation_error(_request: Request, error: RequestValidationError):
    detail = [{key: issue[key] for key in ("loc", "msg", "type")} for issue in error.errors()]
    return JSONResponse(status_code=422, content={"detail": detail})


@lru_cache(maxsize=1)
def load_artifact(path: Path, _file_version: tuple[int, int, int]) -> ModelArtifact:
    artifact = load(path, trusted=TRUSTED_MODEL_TYPES)
    pipeline = artifact["pipeline"]
    if not isinstance(pipeline, Pipeline) or not isinstance(
        pipeline.named_steps.get("clf"), RandomForestClassifier
    ):
        raise ValueError("El artefacto no contiene el pipeline Random Forest.")
    bounds = artifact["feature_bounds"]
    if set(bounds) != set(FEATURES):
        raise ValueError("El artefacto no contiene los rangos del entrenamiento.")
    return ModelArtifact(pipeline, ModelMetrics.model_validate(artifact["metrics"]), bounds)


def model_artifact() -> ModelArtifact | None:
    try:
        version = ARTIFACT_PATH.stat()
        return load_artifact(ARTIFACT_PATH, (version.st_ino, version.st_size, version.st_mtime_ns))
    except FileNotFoundError:
        return None
    except Exception:
        logger.exception("No se pudo cargar el modelo en %s", ARTIFACT_PATH)
        return None


def descending_ranks(cases: list[dict], field: str) -> dict[str, int]:
    ordered = sorted(cases, key=lambda case: (-case[field], case["id_deudor"]))
    return {case["id_deudor"]: rank for rank, case in enumerate(ordered, start=1)}


@app.get("/", response_class=FileResponse, include_in_schema=False)
def index():
    return FileResponse(FRONTEND / "index.html")


app.mount("/static", StaticFiles(directory=FRONTEND), name="static")


@app.get("/api/status", response_model=ModelStatus)
def status():
    artifact = model_artifact()
    return ModelStatus(ready=artifact is not None, metrics=artifact.metrics if artifact else None)


@app.get("/api/sample", response_model=CaseInput)
def sample():
    return {"cases": sample_records()}


@app.post("/api/analyze", response_model=CaseAnalysis)
def analyze(batch: CaseInput):
    active = [case.model_dump() for case in batch.cases if case.situacion_adeudo == 1]
    excluded = [case for case in batch.cases if case.situacion_adeudo == 0]
    if not active:
        return CaseAnalysis(cases=[], excluded=excluded, warnings=[])

    artifact = model_artifact()
    if artifact is None:
        raise HTTPException(
            status_code=503,
            detail="El modelo no está disponible. Genera el artefacto con python -m backend.train.",
        )
    features = prepare_features(pd.DataFrame.from_records(active))
    warnings = []
    probabilities = artifact.pipeline.predict_proba(features)[:, 1]
    below_range = features.lt({field: limit[0] for field, limit in artifact.feature_bounds.items()})
    above_range = features.gt({field: limit[1] for field, limit in artifact.feature_bounds.items()})
    out_of_range = below_range | above_range
    for case, probability, flags in zip(active, probabilities, out_of_range.to_numpy(), strict=True):
        case["probability"] = float(probability)
        fields = out_of_range.columns[flags].tolist()
        if fields:
            warnings.append(OutOfRange(id_deudor=case["id_deudor"], fields=fields))

    rank_model = descending_ranks(active, "probability")
    rank_amount = descending_ranks(active, "monto_pendiente")
    rank_days = descending_ranks(active, "dias_vencimiento")
    for case in active:
        case_id = case["id_deudor"]
        case["rank_model"] = rank_model[case_id]
        case["rank_amount"] = rank_amount[case_id]
        case["rank_days"] = rank_days[case_id]
    active.sort(key=lambda case: case["rank_model"])
    return {"cases": active, "excluded": excluded, "warnings": warnings}
