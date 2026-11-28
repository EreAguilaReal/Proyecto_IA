import json
import shutil

from fastapi.testclient import TestClient
import pytest

from backend import main
from backend.data import INPUT_COLUMNS, generate_records
from backend.train import train
from backend.model import prepare_features
from sklearn.model_selection import train_test_split


client = TestClient(main.app)


def test_evaluation_compares_same_held_out_cases(trained_artifact, monkeypatch):
    monkeypatch.setattr(main, "ARTIFACT_PATH", trained_artifact[0])
    active = generate_records().query('situacion_adeudo == 1')
    features = prepare_features(active)
    _, test_x, _, _ = train_test_split(features, active.recuperado, test_size=.2, random_state=42, stratify=active.recuperado)
    holdout = active.loc[test_x.index].copy()
    holdout['score'] = main.model_artifact().pipeline.predict_proba(test_x)[:, 1]
    response = client.get('/api/evaluation')
    assert response.status_code == 200
    result = response.json()
    assert result['test_cases'] == 897 and result['top_n'] == 50
    for key, column in [('modelo', 'score'), ('monto', 'monto_pendiente'), ('dias', 'dias_vencimiento')]:
        chosen = holdout.sort_values([column, 'id_deudor'], ascending=[False, True]).head(50)
        assert result['recovered'][key] == int(chosen.recuperado.sum())


@pytest.fixture(scope="module")
def trained_artifact(tmp_path_factory):
    path = tmp_path_factory.mktemp("training") / "modelo.skops"
    metrics = train(path)
    return path, metrics


@pytest.fixture
def active_case():
    return next(case for case in client.get("/api/sample").json()["cases"]
                if case["situacion_adeudo"] == 1)


def test_artifact_required_for_active_cases(tmp_path, monkeypatch, active_case):
    monkeypatch.setattr(main, "ARTIFACT_PATH", tmp_path / "missing.skops")
    assert client.get("/api/status").json() == {
        "ready": False, "model": "Random Forest", "data_source": "synthetic", "metrics": None,
    }
    unavailable = client.post("/api/analyze", json={"cases": [active_case]})
    assert unavailable.status_code == 503
    assert "python -m backend.train" in unavailable.json()["detail"]

    excluded = {**active_case, "situacion_adeudo": 0}
    assert client.post("/api/analyze", json={"cases": [excluded]}).json() == {
        "cases": [], "excluded": [excluded], "warnings": [],
    }


def test_sample_is_repeatable_and_independent():
    sample = client.get("/api/sample").json()["cases"]
    assert len(sample) == 30
    assert sample == client.get("/api/sample").json()["cases"]
    assert {case["situacion_adeudo"] for case in sample} == {0, 1}
    assert all(set(case) == set(INPUT_COLUMNS) for case in sample)
    training = generate_records()
    assert set(training["id_deudor"]).isdisjoint(case["id_deudor"] for case in sample)
    feature_rows = set(training[[*INPUT_COLUMNS[1:]]].itertuples(index=False, name=None))
    assert all(tuple(case[column] for column in INPUT_COLUMNS[1:]) not in feature_rows
               for case in sample)


def test_analyze_ranks_active_cases_and_returns_excluded(trained_artifact, monkeypatch):
    path, metrics = trained_artifact
    monkeypatch.setattr(main, "ARTIFACT_PATH", path)
    status = client.get("/api/status").json()
    assert status["ready"] is True
    assert status["metrics"] == metrics

    sample = client.get("/api/sample").json()["cases"]
    response = client.post("/api/analyze", json={"cases": sample})
    assert response.status_code == 200
    output = response.json()
    active = output["cases"]
    assert output["excluded"] == [case for case in sample if case["situacion_adeudo"] == 0]
    assert {case["id_deudor"] for case in active} == {
        case["id_deudor"] for case in sample if case["situacion_adeudo"] == 1
    }
    assert all(0 <= case["probability"] <= 1 for case in active)
    assert active == sorted(active, key=lambda case: case["rank_model"])
    for field, rank in (
        ("probability", "rank_model"), ("monto_pendiente", "rank_amount"),
        ("dias_vencimiento", "rank_days"),
    ):
        ordered = sorted(active, key=lambda case: (-case[field], case["id_deudor"]))
        assert [case[rank] for case in ordered] == list(range(1, len(active) + 1))


def test_out_of_range_days_are_scored_and_reported(trained_artifact, monkeypatch, active_case):
    monkeypatch.setattr(main, "ARTIFACT_PATH", trained_artifact[0])
    case = {**active_case, "dias_vencimiento": 400}
    response = client.post("/api/analyze", json={"cases": [case]})
    assert response.status_code == 200
    assert response.json()["cases"][0]["dias_vencimiento"] == 400
    assert response.json()["warnings"] == [
        {"id_deudor": case["id_deudor"], "fields": ["dias_vencimiento"]},
    ]


def test_model_replacement_is_detected(trained_artifact, tmp_path, monkeypatch):
    path = tmp_path / "modelo.skops"
    shutil.copyfile(trained_artifact[0], path)
    monkeypatch.setattr(main, "ARTIFACT_PATH", path)
    assert client.get("/api/status").json()["ready"] is True

    replacement = tmp_path / "replacement.skops"
    replacement.write_bytes(b"invalid")
    replacement.replace(path)
    assert client.get("/api/status").json()["ready"] is False


@pytest.mark.parametrize(("field", "value"), [
    ("id_deudor", " "),
    ("monto_pendiente", -10),
    ("monto_pendiente", "10.5"),
    ("dias_vencimiento", 3.5),
    ("intentos_contacto", True),
    ("respuesta_contacto", 3),
    ("situacion_adeudo", 2),
    ("recuperado", 1),
])
def test_invalid_values_report_row_and_column(active_case, field, value):
    invalid = {**active_case, field: value}
    response = client.post("/api/analyze", json={"cases": [invalid]})
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "cases", 0, field]


def test_nonfinite_json_numbers_report_validation_error(active_case):
    case = {**active_case, "monto_pendiente": float("inf")}
    response = client.post(
        "/api/analyze",
        content=json.dumps({"cases": [case]}),
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "cases", 0, "monto_pendiente"]


def test_rejects_duplicate_ids_and_incoherent_cases(active_case):
    duplicate = client.post("/api/analyze", json={"cases": [active_case, active_case]})
    assert duplicate.status_code == 422
    assert "duplicado" in duplicate.json()["detail"][0]["msg"]

    empty_amount = {**active_case, "monto_pendiente": 0, "pagos_parciales": 0}
    assert client.post("/api/analyze", json={"cases": [empty_amount]}).status_code == 422

    impossible_contact = {**active_case, "intentos_contacto": 0, "respuesta_contacto": 2}
    inconsistent = client.post("/api/analyze", json={"cases": [impossible_contact]})
    assert inconsistent.status_code == 422
    assert "cero intentos" in inconsistent.json()["detail"][0]["msg"]
