from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ScratchModelSpec:
    vocab_size: int = 32_000
    hidden_size: int = 384
    intermediate_size: int = 1_024
    num_hidden_layers: int = 10
    num_attention_heads: int = 6
    num_key_value_heads: int = 3
    max_position_embeddings: int = 512
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10_000.0
    tie_word_embeddings: bool = True

    @classmethod
    def from_mapping(cls, values: dict[str, Any] | None) -> "ScratchModelSpec":
        spec = cls()
        if not values:
            return spec
        data = asdict(spec)
        for key in data:
            if key in values:
                data[key] = values[key]
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def build_config(self):
        try:
            from transformers import LlamaConfig
        except ImportError as exc:  # pragma: no cover - exercised in CUDA environment
            raise RuntimeError("The cuda extra is required to construct the scratch model.") from exc
        return LlamaConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            max_position_embeddings=self.max_position_embeddings,
            rms_norm_eps=self.rms_norm_eps,
            rope_theta=self.rope_theta,
            hidden_act="silu",
            attention_bias=False,
            attention_dropout=0.0,
            tie_word_embeddings=self.tie_word_embeddings,
            use_cache=True,
            bos_token_id=2,
            eos_token_id=3,
            pad_token_id=0,
        )

    def parameter_count(self) -> int:
        config = self.build_config()
        try:
            from transformers import LlamaForCausalLM
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("The cuda extra is required to construct the scratch model.") from exc
        return sum(parameter.numel() for parameter in LlamaForCausalLM(config).parameters())
