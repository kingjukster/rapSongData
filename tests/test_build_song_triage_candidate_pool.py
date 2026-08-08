from scripts.build_song_triage_candidate_pool import family_memberships, rank_key


def row(*tags: str, decision: str = "keep", confidence: float = 0.8, quality_floor: int = 4):
    return {
        "song_key": "1",
        "decision": decision,
        "tags": list(tags),
        "confidence": confidence,
        "quality_floor": quality_floor,
        "rank": 10,
    }


def test_technical_requires_rhyme_and_cadence() -> None:
    assert "technical" in family_memberships(row("technical_rhyme", "good_cadence"))
    assert "technical" not in family_memberships(row("technical_rhyme"))


def test_clean_requires_safety_and_quality_support() -> None:
    assert "clean" in family_memberships(row("clean_candidate", "vivid_imagery"))
    assert "clean" not in family_memberships(
        row("clean_candidate", "vivid_imagery", "unsafe_clean_training")
    )


def test_hard_and_quality_failures_exclude_all_families() -> None:
    assert not family_memberships(row("technical_rhyme", "good_cadence", "scrape_artifact"))
    assert not family_memberships(row("coherent_story", "vivid_imagery", "generic"))
    assert not family_memberships(row("technical_rhyme", "good_cadence", decision="uncertain"))


def test_rank_prefers_quality_then_confidence() -> None:
    high_floor = row("good_cadence", quality_floor=4, confidence=0.6)
    high_confidence = row("good_cadence", quality_floor=3, confidence=0.99)
    assert rank_key(high_floor) < rank_key(high_confidence)
