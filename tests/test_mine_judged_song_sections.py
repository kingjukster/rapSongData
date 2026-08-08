from scripts.mine_judged_song_sections import (
    clean_safe,
    local_metrics,
    minhash_signature,
    minhash_similarity,
    parent_eligible,
    split_boundary_sections,
)


def test_boundaries_preserve_source_line_numbers() -> None:
    lyrics = "[Verse 1]\n" + "\n".join(f"line {i}" for i in range(1, 13)) + "\n\n[Hook]\nrepeat"
    sections = split_boundary_sections(lyrics)
    assert sections[0]["kind"] == "verse"
    assert len(sections[0]["lines"]) == 12
    assert sections[0]["lines"][0][0] == 2
    assert sections[1]["kind"] == "hook"


def test_clean_safety_is_conservative() -> None:
    safe = local_metrics([f"city window number {i} glows tonight" for i in range(12)])
    unsafe = local_metrics(["clean line here"] * 11 + ["selling cocaine tonight"])
    assert clean_safe(safe)
    assert not clean_safe(unsafe)


def test_parent_eligibility_uses_scores_not_only_full_song_pass() -> None:
    row = {
        "strict_pass": False, "technical_rhyme": 4, "critical_failure_flags": ["weak_ending"],
        "overall": 3, "flow_cadence": 3, "coherence": 2, "thematic_depth": 2,
        "ending_strength": 2, "family_compliance": 3,
    }
    assert parent_eligible(row, "technical")


def test_story_parent_eligibility_requires_strong_narrative_dimensions() -> None:
    row = {
        "strict_pass": False, "critical_failure_flags": [], "truncated": False,
        "overall": 3, "coherence": 4, "thematic_depth": 4, "imagery": 3,
        "ending_strength": 4, "genericness": 3,
    }
    assert parent_eligible(row, "story")

    row["coherence"] = 3
    assert not parent_eligible(row, "story")
    row["coherence"] = 4
    row["critical_failure_flags"] = ["generic_filler"]
    assert not parent_eligible(row, "story")


def test_minhash_detects_identical_text() -> None:
    text = "\n".join(f"the city light keeps moving number {i}" for i in range(12))
    signature = minhash_signature(text)
    assert minhash_similarity(signature, signature) == 1.0
