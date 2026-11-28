import json
from pathlib import Path
from tempfile import NamedTemporaryFile

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from skops.io import dump

from .data import generate_records
from .model import ARTIFACT_PATH, make_pipeline, prepare_features


def train(output: Path = ARTIFACT_PATH) -> dict:
    records = generate_records()
    active = records[records["situacion_adeudo"] == 1]
    features = prepare_features(active)
    target = active["recuperado"]
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.20, random_state=42, stratify=target
    )
    classifiers = {
        "naive_bayes": GaussianNB(),
        "random_forest": RandomForestClassifier(
            n_estimators=150, max_depth=8, class_weight="balanced", random_state=42
        ),
    }
    pipelines = {}
    metrics = {}
    for name, classifier in classifiers.items():
        pipeline = make_pipeline(classifier).fit(x_train, y_train)
        pipelines[name] = pipeline
        metrics[name] = {
            "f1": float(f1_score(y_test, pipeline.predict(x_test))),
            "roc_auc": float(roc_auc_score(y_test, pipeline.predict_proba(x_test)[:, 1])),
        }

    output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=output.parent, suffix=".skops", delete=False) as temporary_file:
        temporary_path = Path(temporary_file.name)
    try:
        bounds = {
            feature: (float(x_train[feature].min()), float(x_train[feature].max()))
            for feature in x_train.columns
        }
        dump({
            "pipeline": pipelines["random_forest"],
            "metrics": metrics,
            "feature_bounds": bounds,
        }, temporary_path)
        temporary_path.replace(output)
    finally:
        temporary_path.unlink(missing_ok=True)
    return metrics


if __name__ == "__main__":
    print(json.dumps(train(), indent=2))
    print(f"Modelo reproducido a partir de datos sinteticos: {ARTIFACT_PATH}")
