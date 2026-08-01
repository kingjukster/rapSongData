"""Quality-max lyric generation and transformation research lane."""

from .prompts import build_messages
from .scoring import clean_lyrics, rank_candidates, score_candidate

__all__ = ["build_messages", "clean_lyrics", "rank_candidates", "score_candidate"]
