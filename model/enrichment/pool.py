try:
    from .pool_common import (
        _PoolImageDataset,
        _PoolTextDataset,
        _pool_transform,
        _unwrap_model,
    )
    from .pool_manager import TargetPoolManager
except ImportError:
    import importlib.util
    import pathlib
    import sys
    import types

    _ROOT = pathlib.Path(__file__).resolve().parent
    _PKG = "_enrichment_pool_compat"
    if _PKG not in sys.modules:
        _package = types.ModuleType(_PKG)
        _package.__path__ = [str(_ROOT)]
        sys.modules[_PKG] = _package

    def _load_sibling(name):
        qualified_name = f"{_PKG}.{name}"
        if qualified_name in sys.modules:
            return sys.modules[qualified_name]
        spec = importlib.util.spec_from_file_location(qualified_name, _ROOT / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    _common = _load_sibling("pool_common")
    _manager = _load_sibling("pool_manager")

    _PoolImageDataset = _common._PoolImageDataset
    _PoolTextDataset = _common._PoolTextDataset
    _pool_transform = _common._pool_transform
    _unwrap_model = _common._unwrap_model
    TargetPoolManager = _manager.TargetPoolManager

__all__ = [
    "TargetPoolManager",
]
