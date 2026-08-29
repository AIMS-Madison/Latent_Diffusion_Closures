"""Neural-network architectures used by the closure models."""

from .autoencoder import FieldAutoencoder
from .diffusion import ConditionalFNOScore

__all__ = ["ConditionalFNOScore", "FieldAutoencoder"]
