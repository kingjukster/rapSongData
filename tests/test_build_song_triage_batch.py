import json

from scripts.build_song_triage_batch import SongRow, approx_tokens, compact_lyrics, parse_response_json, request_body


def test_compact_lyrics_preserves_line_breaks_and_reports_truncation() -> None:
    text, truncated = compact_lyrics("one\n\n two two\nthree three three", max_chars=14)

    assert text == "one\ntwo two"
    assert truncated is True


def test_request_body_uses_tiny_structured_output() -> None:
    song = SongRow(
        rank=7,
        song_key="abc123",
        title="Example",
        artist="Tester",
        year=2026,
        views=100,
        lyrics="line one\nline two",
    )

    body, source = request_body(song, model="gpt-5.4-nano", max_chars=200, max_output_tokens=120)

    assert body["model"] == "gpt-5.4-nano"
    assert body["reasoning"] == {"effort": "low"}
    assert body["max_output_tokens"] == 120
    assert body["text"]["format"]["type"] == "json_schema"
    assert body["text"]["format"]["schema"]["properties"]["decision"]["enum"] == ["keep", "reject", "uncertain"]
    assert source["custom_id"] == "songtriage:abc123"
    assert source["rank"] == 7
    assert source["truncated"] is False

    user_payload = json.loads(body["input"][1]["content"].split("\n", 1)[1])
    assert user_payload["songs"][0]["lyrics"] == "line one\nline two"


def test_approx_tokens_is_conservative_for_words_and_chars() -> None:
    assert approx_tokens("word " * 100) >= 135
    assert approx_tokens("x" * 360) >= 100


def test_parse_response_json_reads_responses_output_text() -> None:
    payload = {
        "output": [
            {"type": "reasoning", "content": []},
            {
                "type": "message",
                "content": [
                    {
                        "type": "output_text",
                        "text": '{"decision":"keep","confidence":0.8,"quality_floor":4,"tags":["strong_candidate"],"reason":"good"}',
                    }
                ],
            },
        ]
    }

    assert parse_response_json(payload)["decision"] == "keep"
