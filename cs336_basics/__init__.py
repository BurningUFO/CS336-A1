import importlib.metadata

from .tokenizer_optimized import Tokenizer, train_bpe

try:
    __version__ = importlib.metadata.version("cs336_basics")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0+local"

__all__ = ["__version__", "Tokenizer", "train_bpe"]
