"""Direction3 memory router package exports with lazy imports."""

from memory_router.feature_schema import FEATURE_ORDER

__all__ = [
    "FEATURE_ORDER",
    "build_datasets",
    "extract_state_features",
    "infer",
    "loss_fn",
    "normalize_mix",
    "RouterMLP",
    "train",
    "vectorize_features",
]


def extract_state_features(*args, **kwargs):
    from memory_router.feature_schema import extract_state_features as _fn

    return _fn(*args, **kwargs)


def vectorize_features(*args, **kwargs):
    from memory_router.feature_schema import vectorize_features as _fn

    return _fn(*args, **kwargs)


def normalize_mix(*args, **kwargs):
    from memory_router.feature_schema import normalize_mix as _fn

    return _fn(*args, **kwargs)


def build_datasets(*args, **kwargs):
    from memory_router.dataset_builder import build_datasets as _fn

    return _fn(*args, **kwargs)


def train(*args, **kwargs):
    from memory_router.train_router import train as _fn

    return _fn(*args, **kwargs)


def infer(*args, **kwargs):
    from memory_router.infer_router import infer as _fn

    return _fn(*args, **kwargs)


def loss_fn(*args, **kwargs):
    from memory_router.model import loss_fn as _fn

    return _fn(*args, **kwargs)


class RouterMLP:  # pragma: no cover
    def __new__(cls, *args, **kwargs):
        from memory_router.model import RouterMLP as _RouterMLP

        return _RouterMLP(*args, **kwargs)
