import json

import pytest
from types import SimpleNamespace

from rap_song_data.evaluation.fixed_generation import (
    generation_eos_token_ids,
    generation_pad_token_id,
    load_prompts,
)


def test_load_prompts_accepts_structured_json(tmp_path):
    path = tmp_path / "prompts.json"
    path.write_text(
        json.dumps([{"prompt": " First prompt ", "theme": "test"}, {"prompt": "Second prompt"}]),
        encoding="utf-8",
    )

    assert load_prompts(path) == ["First prompt", "Second prompt"]


def test_load_prompts_rejects_structured_json_without_prompt(tmp_path):
    path = tmp_path / "prompts.json"
    path.write_text(json.dumps([{"theme": "test"}]), encoding="utf-8")

    with pytest.raises(ValueError, match="string 'prompt' field"):
        load_prompts(path)


def test_generation_tokens_honor_model_specific_multi_eos_and_tokenizer_pad():
    model = SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=[100265, 100257]))
    tokenizer = SimpleNamespace(eos_token_id=100257, pad_token_id=100277)
    assert generation_eos_token_ids(model, tokenizer) == [100265, 100257]
    assert generation_pad_token_id(tokenizer) == 100277
