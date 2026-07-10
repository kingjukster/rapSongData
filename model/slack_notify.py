"""Legacy wrapper for :mod:`rap_song_data.integrations.slack`."""

try:
    from ._bootstrap import add_src_to_path
except ImportError:
    from _bootstrap import add_src_to_path

add_src_to_path()

from rap_song_data.integrations.slack import *  # noqa: F401,F403,E402
