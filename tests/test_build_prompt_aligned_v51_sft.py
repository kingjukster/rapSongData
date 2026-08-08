from collections import Counter

from scripts.build_prompt_aligned_v51_sft import aligned_prompt, note_descriptor, topic_keywords


def test_keywords_prefer_specific_repeated_terms() -> None:
    text = "Lantern by the flooded doorway\nThe lantern shakes while basement water climbs"
    keywords = topic_keywords(text, Counter({"lantern": 1, "flooded": 1, "doorway": 1, "basement": 1, "water": 1, "climbs": 1}), 10)
    assert keywords[0] == "lantern"
    assert "basement" in keywords


def test_family_prompt_includes_descriptor() -> None:
    prompt = aligned_prompt("story", ["lantern", "basement", "water"])
    assert "lantern, basement, water" in prompt
    assert "exactly 12" in prompt


def test_note_descriptor_keeps_semantic_first_sentence() -> None:
    note = "A self-contained reflection on rent, family pressure, and pride. Cadence is strong."
    assert note_descriptor(note) == "A self-contained reflection on rent, family pressure, and pride"
