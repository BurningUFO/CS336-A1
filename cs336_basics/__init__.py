import importlib.metadata

from .tokenizer_optimized import Tokenizer, train_bpe

__version__ = importlib.metadata.version("cs336_basics")

__all__ = ["__version__", "Tokenizer", "train_bpe"]
