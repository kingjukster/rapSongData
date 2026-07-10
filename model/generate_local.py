"""Legacy wrapper for :mod:`rap_song_data.generation.local`."""

try:
    from ._bootstrap import add_src_to_path
except ImportError:
    from _bootstrap import add_src_to_path

add_src_to_path()

from rap_song_data.generation.local import *  # noqa: F401,F403,E402
from rap_song_data.generation.local import main  # noqa: E402


if __name__ == "__main__":
    main()
