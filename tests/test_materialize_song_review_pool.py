from scripts.materialize_song_review_pool import normalized_hash


def test_normalized_hash_ignores_whitespace_and_case() -> None:
    assert normalized_hash("ONE\n two") == normalized_hash(" one   TWO ")
