from scripts.build_section_mined_v4_sft import choose_unique_sources


def candidate(song: str, family: str, score: int = 4) -> dict:
    return {
        "section_id": f"{song}-{family}", "song_key": song, "family": family,
        "computed_strict_pass": True, "critical_failure_flags": ["none"],
        "text": "\n".join(f"line {index}" for index in range(12)),
        "overall": score, "technical_rhyme": score, "genericness": 2,
        "ending_strength": score, "coherence": score, "thematic_specificity": score,
        "natural_phrasing_cadence": score, "local_score": float(score),
    }


def test_unique_source_selection_and_exclusion() -> None:
    rows = [candidate("1", "technical", 5), candidate("1", "technical", 4), candidate("2", "technical", 4)]
    chosen = choose_unique_sources(rows, "technical", 1, {"1"})
    assert [row["song_key"] for row in chosen] == ["2"]
