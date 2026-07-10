from __future__ import annotations

import unittest

from scripts.build_smoke_eval_followup_reports import select_pair_buckets, summarize_latency_run


def ranked_row(row_id: str, score: float) -> dict:
    return {
        "row_id": row_id,
        "candidate_index": int(row_id[1:]),
        "sample_index": 0,
        "prompt_key": f"prompt-{row_id}",
        "prompt": "Write exactly 12 lines.",
        "theme": "test theme",
        "style": "test style",
        "lyrics": "one\ntwo",
        "quality_score": score,
        "quality_tags": [],
        "quality_dimensions": {"rhyme_density": score, "specific_imagery": score / 2},
    }


class SmokeEvalFollowupReportTests(unittest.TestCase):
    def test_pair_bucket_selection_is_disjoint(self) -> None:
        base = [ranked_row(f"r{i}", 0.5) for i in range(1, 8)]
        adapter_scores = [0.9, 0.8, 0.51, 0.5, 0.49, 0.2, 0.1]
        adapter = [ranked_row(f"r{i}", score) for i, score in enumerate(adapter_scores, 1)]

        selected, summary = select_pair_buckets(base, adapter, pairs_per_bucket=2)
        ids = [row["row_id"] for row in selected]

        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(summary["paired_rows"], 7)
        self.assertEqual(summary["selected_counts"]["adapter_large_win"], 2)
        self.assertEqual(summary["selected_counts"]["base_large_win"], 2)
        self.assertEqual(summary["selected_counts"]["close_or_tie"], 2)

    def test_latency_summary_uses_batch_chunks_and_retry_attempts(self) -> None:
        summary = {
            "generation_wall_seconds": 12.0,
            "load_seconds": 1.0,
            "underlength_retry_triggered": 1,
            "underlength_retry_generations": 1,
            "settings": {"adapter_enabled": False},
            "runtime": {"bf16_supported": True},
        }
        rows = [
            {
                "candidate_index": 1,
                "raw_line_count": 12,
                "postprocessed_line_count": 12,
                "postprocess_applied": False,
                "timing": {"batch_size": 2, "batch_seconds": 4.0, "effective_generated_tokens": 20},
                "retry_attempts": [],
            },
            {
                "candidate_index": 2,
                "raw_line_count": 12,
                "postprocessed_line_count": 12,
                "postprocess_applied": False,
                "timing": {"batch_size": 2, "batch_seconds": 4.0, "effective_generated_tokens": 20},
                "retry_attempts": [],
            },
            {
                "candidate_index": 3,
                "raw_line_count": 12,
                "postprocessed_line_count": 12,
                "postprocess_applied": False,
                "timing": {"batch_size": 2, "batch_seconds": 5.0, "effective_generated_tokens": 20},
                "retry_attempts": [],
            },
            {
                "candidate_index": 4,
                "raw_line_count": 12,
                "postprocessed_line_count": 12,
                "postprocess_applied": False,
                "timing": {"batch_size": 1, "batch_seconds": 2.0, "effective_generated_tokens": 10},
                "retry_attempts": [{"timing": {"batch_seconds": 2.0}}],
            },
        ]

        result = summarize_latency_run("base", summary, rows)

        self.assertEqual(result["configured_or_effective_batch_size"], 2)
        self.assertEqual(result["estimated_initial_batch_count"], 2)
        self.assertEqual(result["estimated_initial_batch_seconds"], 9.0)
        self.assertEqual(result["retry_seconds"], 2.0)
        self.assertEqual(result["rows_per_minute"], 20.0)


if __name__ == "__main__":
    unittest.main()
