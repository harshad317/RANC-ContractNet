import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline

from ranc_contractnet import RANCDataTransformer


def test_fit_statistics_do_not_include_test_split():
    X_train = np.array([[0.0], [1.0], [2.0], [3.0]])
    X_test = np.array([[10_000.0]])
    transformer = RANCDataTransformer(
        contracts={"global": {"hard_clauses": {"enforce_scale_invariance": True}}},
        random_state=0,
    ).fit(X_train)
    card = transformer.cards_[0]
    assert card.max == 3.0
    assert card.n_samples == 4
    transformed_test = transformer.transform(X_test)
    assert transformed_test.shape == (1, 1)


def test_sklearn_pipeline_cross_validation_runs_without_global_fit():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 4))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    pipe = Pipeline(
        [
            ("ranc", RANCDataTransformer(random_state=0)),
            ("model", LogisticRegression(max_iter=200)),
        ]
    )
    scores = cross_val_score(pipe, X, y, cv=KFold(n_splits=3, shuffle=True, random_state=0))
    assert scores.shape == (3,)
    assert np.all(np.isfinite(scores))

