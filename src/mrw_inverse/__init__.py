"""Physics-constrained multiscale inverse framework for stochastic signals."""

from .data import (
    AntiConfoundedConfig,
    DEFAULT_Q_GRID,
    PROCESS_TYPES,
    SpectralRepresentationDatasetConfig,
    generate_anti_confounded_dataset,
    generate_process_sample,
    generate_spectral_representation_dataset,
)
from .models.cmin import CMINRegressor
from .models.mrw_decoder import MRWAnalyticDecoder
from .models.spectral_representation_model import CMINSRModel, SpectralRepresentationModel
from .models.empirical_spectrum import estimate_empirical_spectrum
from .proxy import estimate_window, proxy_inverse_estimate
from .surrogates import block_shuffled_surrogate, phase_randomized_surrogate, shuffled_surrogate

__all__ = [
    "AntiConfoundedConfig",
    "DEFAULT_Q_GRID",
    "PROCESS_TYPES",
    "SpectralRepresentationDatasetConfig",
    "generate_anti_confounded_dataset",
    "generate_process_sample",
    "generate_spectral_representation_dataset",
    "CMINRegressor",
    "CMINSRModel",
    "MRWAnalyticDecoder",
    "SpectralRepresentationModel",
    "estimate_empirical_spectrum",
    "estimate_window",
    "proxy_inverse_estimate",
    "shuffled_surrogate",
    "block_shuffled_surrogate",
    "phase_randomized_surrogate",
]
