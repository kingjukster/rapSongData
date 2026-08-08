from scripts.build_section_quality_judge_batch import computed_strict_pass, request_body


def passing(family: str) -> dict:
    return {
        "family": family, "overall": 4, "coherence": 4, "thematic_specificity": 4,
        "ending_strength": 4, "natural_phrasing_cadence": 4, "technical_rhyme": 4,
        "genericness": 2, "safety": 5, "self_contained_excerpt": 4,
        "critical_failure_flags": ["none"],
    }


def test_family_gates() -> None:
    assert computed_strict_pass(passing("technical"))
    assert computed_strict_pass(passing("clean"))
    story = passing("story")
    story["safety"] = 3
    assert computed_strict_pass(story)
    story["genericness"] = 3
    assert not computed_strict_pass(story)
    row = passing("clean")
    row["genericness"] = 3
    assert not computed_strict_pass(row)


def test_prompt_explicitly_says_excerpt() -> None:
    body = request_body({"section_id": "s1", "family": "technical", "text": "x\n" * 12}, "gpt-5.5", 650)
    assert "excerpt, not a complete song" in body["input"][0]["content"]
    assert body["reasoning"] == {"effort": "low"}


def test_story_prompt_uses_story_specific_gate() -> None:
    body = request_body({"section_id": "s1", "family": "story", "text": "x\n" * 12}, "gpt-5.5", 650)
    assert "For story excerpts" in body["input"][0]["content"]
