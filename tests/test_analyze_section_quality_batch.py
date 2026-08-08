from scripts.analyze_section_quality_batch import response_json


def test_recovers_complete_gate_fields_when_only_notes_are_truncated() -> None:
    text = (
        '{"section_id":"s1","overall":4,"coherence":4,"thematic_specificity":4,'
        '"ending_strength":4,"natural_phrasing_cadence":4,"technical_rhyme":3,'
        '"genericness":2,"safety":3,"self_contained_excerpt":4,'
        '"critical_failure_flags":["none"],"strict_pass":true,'
        '"notes":"unfinished'
    )
    row = response_json({"output_text": text})
    assert row["strict_pass"] is True
    assert row["parse_recovered_truncated_notes"] is True


def test_does_not_recover_incomplete_gate_fields() -> None:
    text = '{"section_id":"s1","overall":4,"notes":"unfinished'
    try:
        response_json({"output_text": text})
    except Exception:
        return
    raise AssertionError("incomplete gate fields must not be recovered")
