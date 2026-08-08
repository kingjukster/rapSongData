from scripts.build_song_quality_judge_batch import compact_lyrics, request_body


def test_compact_lyrics_preserves_complete_lines() -> None:
    text, truncated = compact_lyrics("one two\nthree four\nfive six", 19)
    assert text == "one two\nthree four"
    assert truncated


def test_request_is_blind_and_family_specific() -> None:
    body, source = request_body(
        {"song_key": "42", "lyrics": "bars here", "judge_family": "clean"},
        "gpt-5.5", 6000, 900,
    )
    user_text = body["input"][1]["content"]
    assert '"song_key":"42"' in user_text
    assert '"target_family":"clean"' in user_text
    assert body["reasoning"] == {"effort": "low"}
    assert source["custom_id"] == "songquality:42"
