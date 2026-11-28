import numpy as np
import pandas as pd


TRAINING_SEED = 42
SAMPLE_SEED = 2026
INPUT_COLUMNS = [
    "id_deudor",
    "monto_pendiente",
    "dias_vencimiento",
    "incumplimientos_anteriores",
    "pagos_parciales",
    "intentos_contacto",
    "respuesta_contacto",
    "situacion_adeudo",
]


def generate_records(
    seed: int = TRAINING_SEED, count: int = 5000, id_prefix: str = "DEU"
) -> pd.DataFrame:
    """Reproduce datos.ipynb without modifying the global random state."""
    rng = np.random.RandomState(seed)
    id_deudor = [f"{id_prefix}-{i:04d}" for i in range(1, count + 1)]
    monto_original = np.round(rng.lognormal(mean=8.5, sigma=0.8, size=count), 2)
    dias_vencimiento = rng.randint(1, 180, size=count)
    incumplimientos = rng.poisson(lam=1.2, size=count)

    tiene_abono = rng.binomial(1, p=0.45, size=count)
    porcentaje_abonado = np.where(
        tiene_abono == 1, rng.uniform(0.1, 0.8, size=count), 0.0
    )
    pagos_parciales = np.round(monto_original * porcentaje_abonado, 2)
    monto_pendiente = np.round(monto_original - pagos_parciales, 2)

    intentos_contacto = rng.randint(0, 12, size=count)
    respuesta_contacto = np.array([
        rng.choice([0, 1, 2], p=[0.45, 0.35, 0.20]) if intentos else 0
        for intentos in intentos_contacto
    ])
    situacion_adeudo = rng.choice([0, 1], size=count, p=[0.10, 0.90])

    score_oculto = (
        -0.015 * dias_vencimiento
        - 0.35 * incumplimientos
        + 1.8 * porcentaje_abonado
        + np.where(respuesta_contacto == 2, 1.5, np.where(respuesta_contacto == 1, 0.3, -0.5))
        - 0.05 * np.maximum(0, intentos_contacto - 5)
        + rng.normal(0, 0.6, size=count)
    )
    recuperado = rng.binomial(1, p=1 / (1 + np.exp(-score_oculto)))
    return pd.DataFrame(dict(zip(
        [*INPUT_COLUMNS, "recuperado"],
        [id_deudor, monto_pendiente, dias_vencimiento, incumplimientos, pagos_parciales,
         intentos_contacto, respuesta_contacto, situacion_adeudo, recuperado],
        strict=True,
    )))


def sample_records() -> list[dict]:
    return generate_records(SAMPLE_SEED, 30, "DEMO")[INPUT_COLUMNS].to_dict("records")
