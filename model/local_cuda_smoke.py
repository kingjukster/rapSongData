"""Legacy wrapper for :mod:`rap_song_data.training.local_cuda_smoke`."""

try:
    from ._bootstrap import add_src_to_path
except ImportError:
    from _bootstrap import add_src_to_path

add_src_to_path()

from rap_song_data.training.local_cuda_smoke import main  # noqa: E402


if __name__ == "__main__":
    main()
