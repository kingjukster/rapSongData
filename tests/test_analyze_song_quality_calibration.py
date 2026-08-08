from scripts.analyze_song_quality_calibration import computed_strict_pass, family_relevant_strict_pass


def passing_row(family: str = "technical") -> dict:
    return {
        "judge_family": family,
        "overall": 4,
        "technical_rhyme": 4,
        "flow_cadence": 4,
        "coherence": 4,
        "thematic_depth": 4,
        "ending_strength": 4,
        "family_compliance": 5,
        "cleanliness": 5,
        "genericness": 2,
        "critical_failure_flags": ["none"],
    }


def test_recomputes_technical_gate() -> None:
    row = passing_row()
    assert computed_strict_pass(row)
    row["flow_cadence"] = 3
    assert not computed_strict_pass(row)


def test_clean_adds_cleanliness_and_genericness() -> None:
    row = passing_row("clean")
    assert computed_strict_pass(row)
    row["genericness"] = 3
    assert not computed_strict_pass(row)


def test_real_failure_flag_blocks_pass() -> None:
    row = passing_row()
    row["critical_failure_flags"] = ["weak_ending"]
    assert not computed_strict_pass(row)


def test_clean_only_safety_flag_does_not_block_technical_relevant_gate() -> None:
    row = passing_row()
    row["critical_failure_flags"] = ["unsafe_for_clean"]
    assert not computed_strict_pass(row)
    assert family_relevant_strict_pass(row)
