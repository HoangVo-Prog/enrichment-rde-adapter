try:
    from .enricher import TargetPrototypeEnricher, _ResidualGateMLP, _masked_logsumexp
    from .mixer import (
        RankPartQueryConditionedMixerAdapter,
        _FusionMLP,
        _RankPartMixerBlock,
        _two_layer_mlp,
    )
    from .prototypes import (
        EXTRACTOR_MODES,
        TARGET_RELATIVE_MODES,
        build_evidence_bank,
        build_part_prototypes,
        evidence_slot_indices,
        finalize_target_evidence_cache,
        prototype_slot_count,
    )
except ImportError:
    import importlib.util
    import pathlib
    import sys
    import types

    _ROOT = pathlib.Path(__file__).resolve().parent
    _PKG = "_enrichment_modules_compat"
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

    _enricher = _load_sibling("enricher")
    _mixer = _load_sibling("mixer")
    _prototypes = _load_sibling("prototypes")

    TargetPrototypeEnricher = _enricher.TargetPrototypeEnricher
    _ResidualGateMLP = _enricher._ResidualGateMLP
    _masked_logsumexp = _enricher._masked_logsumexp
    RankPartQueryConditionedMixerAdapter = _mixer.RankPartQueryConditionedMixerAdapter
    _FusionMLP = _mixer._FusionMLP
    _RankPartMixerBlock = _mixer._RankPartMixerBlock
    _two_layer_mlp = _mixer._two_layer_mlp
    EXTRACTOR_MODES = _prototypes.EXTRACTOR_MODES
    TARGET_RELATIVE_MODES = _prototypes.TARGET_RELATIVE_MODES
    build_evidence_bank = _prototypes.build_evidence_bank
    build_part_prototypes = _prototypes.build_part_prototypes
    evidence_slot_indices = _prototypes.evidence_slot_indices
    finalize_target_evidence_cache = _prototypes.finalize_target_evidence_cache
    prototype_slot_count = _prototypes.prototype_slot_count

__all__ = [
    "EXTRACTOR_MODES",
    "RankPartQueryConditionedMixerAdapter",
    "TARGET_RELATIVE_MODES",
    "TargetPrototypeEnricher",
    "build_evidence_bank",
    "build_part_prototypes",
    "evidence_slot_indices",
    "finalize_target_evidence_cache",
    "prototype_slot_count",
]
