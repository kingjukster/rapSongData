"""Legacy wrapper for :mod:`rap_song_data.datasets.training_data`."""

try:
    from ._bootstrap import add_src_to_path
except ImportError:  # Direct ``python model/build_training_data.py`` execution.
    from _bootstrap import add_src_to_path

add_src_to_path()

from rap_song_data.datasets.training_data import *  # noqa: F401,F403,E402
from rap_song_data.datasets.training_data import main  # noqa: E402


if __name__ == "__main__":
    main()
