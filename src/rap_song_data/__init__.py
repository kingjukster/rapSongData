"""Tools for curating rap corpora and training lyric-generation models."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("rap-song-data")
except PackageNotFoundError:  # Running directly from a source checkout.
    __version__ = "0.1.0"

__all__ = ["__version__"]
