# lab2/text_recognizer/__init__.py
# Keep imports lazy/lightweight; submodules will be imported where needed.
from . import data, models, lit_models

__all__ = ["data", "models", "lit_models"]
