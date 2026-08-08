from scripts.build_section_mined_v5_sft import choose_preferred_unique_sources


def candidate(song: str, family: str, score: int = 4) -> dict:
    return {
        "section_id": f"{song}-{family}", "song_key": song, "family": family,
        "computed_strict_pass": True, "critical_failure_flags": ["none"],
        "text": "\n".join(f"line {index}" for index in range(12)),
        "overall": score, "technical_rhyme": score, "genericness": 2,
        "ending_strength": score, "coherence": score, "thematic_specificity": score,
        "natural_phrasing_cadence": score, "local_score": float(score),
    }


def test_preferred_selection_avoids_shared_sources_first() -> None:
    rows = [candidate("shared", "story", 5), candidate("unique", "story", 4)]
    chosen = choose_preferred_unique_sources(rows, "story", 1, set(), {"shared"})
    assert chosen[0]["song_key"] == "unique"
