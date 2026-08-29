"""Irregular-mesh neural operator models."""

import importlib

# Heavy baseline models (e.g. UPT -> kappamodules) are imported lazily so that
# using one model does not require every optional dependency of the others.
_MODULE_OF = {
    "FNO": "fno",
    "FNOLoss": "fno",
    "FusionDeepONet": "fusion_deeponet",
    "FusionDeepONetLoss": "fusion_deeponet",
    "GNOT": "gnot",
    "GNOTLoss": "gnot",
    "PCNO": "pcno",
    "PCNOLoss": "pcno",
    "build_aux_from_pos": "pcno",
    "collate_aux_batch": "pcno",
    "compute_fourier_modes": "pcno",
    "PointTokenLoss": "tokenizer_ablation",
    "PointTokenOperator": "tokenizer_ablation",
    "Transolver": "transolver",
    "TransolverLoss": "transolver",
    "UPT": "upt",
    "UPTLoss": "upt",
}


def __getattr__(name):
    module_name = _MODULE_OF.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(f".{module_name}", __name__)
    return getattr(module, name)


def __dir__():
    return sorted(_MODULE_OF)


__all__ = sorted(_MODULE_OF)
