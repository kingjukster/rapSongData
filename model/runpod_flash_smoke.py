"""Legacy wrapper for :mod:`rap_song_data.training.runpod_flash_smoke`."""

import asyncio

try:
    from ._bootstrap import add_src_to_path
except ImportError:
    from _bootstrap import add_src_to_path

add_src_to_path()

from rap_song_data.training.runpod_flash_smoke import *  # noqa: F401,F403,E402
from rap_song_data.training.runpod_flash_smoke import main  # noqa: E402


if __name__ == "__main__":
    asyncio.run(main())
