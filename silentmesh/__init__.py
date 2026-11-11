"""silentmesh: traffic-shape morphing against traffic analysis."""

from .audit import accuracy, audit, epsilon_estimate, train_classifier
from .cover import CoverModel, Packet, default_video_model, generate_cover_traffic
from .morph import demorph, morph_message, unmorphed_stream

__all__ = [
    "CoverModel",
    "Packet",
    "accuracy",
    "audit",
    "default_video_model",
    "demorph",
    "epsilon_estimate",
    "generate_cover_traffic",
    "morph_message",
    "train_classifier",
    "unmorphed_stream",
]

__version__ = "0.1.0"
