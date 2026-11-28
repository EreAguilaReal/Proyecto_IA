import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.model_selection import train_test_split
from skops.io import dump, load

from backend import main
from backend.data import generate_records
from backend.model import EXPLANATION_GROUPS, prepare_features
from backend.tests.test_api import trained_artifact, active_case

client = TestClient(main.app)


def test_explanation_matches_direct_predictions_and_training_references(trained_artifact, monkeypatch, active_case):
    monkeypatch.setattr(main, 'ARTIFACT_PATH', trained_artifact[0])
    response = client.post('/api/explain', json={'case': active_case})
    assert response.status_code == 200
    result = response.json()
    pipeline = main.model_artifact().pipeline
    active = generate_records().query('situacion_adeudo == 1')
    features = prepare_features(active)
    x_train, _, _, _ = train_test_split(features, active.recuperado, test_size=.2, random_state=42, stratify=active.recuperado)
    def score(row):
        return pipeline.predict_proba(prepare_features(pd.DataFrame([row])))[0, 1]
    assert result['score'] == pytest.approx(score(active_case))
    baseline = dict(active_case)
    for item, (_, (label, fields)) in zip(result['items'], EXPLANATION_GROUPS.items(), strict=True):
        assert item['label'] == label
        assert item['original'] == {field: active_case[field] for field in fields}
        reference = item['reference']
        assert ((x_train[list(fields)] == pd.Series(reference)).all(axis=1)).any()
        counterfactual = {**active_case, **reference}
        assert item['delta'] == pytest.approx((score(active_case) - score(counterfactual)) * 100)
        baseline.update(reference)
    assert result['baseline_score'] == pytest.approx(score(baseline))
    assert 'no causalidad' in result['method'] and 'no observadas' in result['method']


def test_legacy_and_bad_references_do_not_break_inference(trained_artifact, tmp_path, monkeypatch, active_case):
    artifact = load(trained_artifact[0], trusted=main.TRUSTED_MODEL_TYPES)
    for reference in (None, {'contacto': {}}, {'adeudo': {}}):
        artifact['explanation_references'] = reference
        path = tmp_path / f'{len(list(tmp_path.iterdir()))}.skops'
        dump(artifact, path)
        monkeypatch.setattr(main, 'ARTIFACT_PATH', path)
        assert client.post('/api/analyze', json={'cases': [active_case]}).status_code == 200
        assert client.post('/api/explain', json={'case': active_case}).status_code == 503


def test_explanation_validation_and_boundaries(trained_artifact, monkeypatch, active_case):
    monkeypatch.setattr(main, 'ARTIFACT_PATH', trained_artifact[0])
    for case in ({**active_case, 'situacion_adeudo': 0},
                 {**active_case, 'monto_pendiente': 0, 'pagos_parciales': 0},
                 {**active_case, 'intentos_contacto': 0, 'respuesta_contacto': 2}):
        assert client.post('/api/explain', json={'case': case}).status_code == 422
    for case in ({**active_case, 'monto_pendiente': 0, 'pagos_parciales': 1, 'intentos_contacto': 0, 'respuesta_contacto': 0},
                 {**active_case, 'monto_pendiente': 1, 'pagos_parciales': 0, 'dias_vencimiento': 0}):
        assert client.post('/api/explain', json={'case': case}).status_code == 200
    assert client.post('/api/explain', json={'case': active_case, 'extra': 1}).status_code == 422
