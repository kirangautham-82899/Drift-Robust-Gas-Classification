"""Helpers for repeating randomised models with different seeds (Step 12)."""


def set_all_seeds(estimator, seed):
    """Set every `random_state` parameter found anywhere inside an estimator (pipelines and the
    stacking classifier included) to the given seed. Returns the estimator."""
    params = {key: seed for key in estimator.get_params(deep=True) if key.endswith("random_state")}
    if params:
        estimator.set_params(**params)
    return estimator
