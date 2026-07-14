from __future__ import annotations

import argparse
import json

from .acquisition import (
    acquire_gutenberg,
    add_gutenberg_arguments,
    add_gutenberg_review_arguments,
    review_gutenberg,
)
from .corpus import add_build_arguments, build_corpus
from .compliance import add_compliance_arguments, analyze_compliance
from .comparison import (
    add_blind_comparison_arguments,
    add_build_comparison_arguments,
    add_scratch_comparison_arguments,
    build_blind_comparison,
    build_comparison,
    generate_scratch_comparison,
)
from .evaluation import add_evaluation_arguments, evaluate
from .release import add_release_arguments, freeze_checkpoint
from .router import add_router_arguments, generate_controlled
from .sweep import add_sweep_arguments, run_sweep
from .source_planning import add_source_planning_arguments, plan_corpus
from .source_governance import (
    add_admit_arguments,
    add_audit_arguments,
    add_ingest_arguments,
    add_inspect_arguments,
    add_materialize_profile_arguments,
    add_migrate_arguments,
    add_profile_arguments,
    add_revoke_arguments,
    add_revocation_impact_arguments,
    add_verify_profile_arguments,
    admit_source,
    audit_source,
    build_profile,
    ingest_source,
    inspect_corpus,
    materialize_profile,
    migrate_source,
    revoke_source,
    revocation_impact,
    verify_profile,
)
from .tokenization import add_tokenizer_arguments, train_and_tokenize
from .training import add_training_arguments, train


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rap-scratch",
        description="Private scratch-model corpus, tokenizer, training, SFT, and evaluation workflow.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    corpus_parser = subparsers.add_parser("build-corpus", help="Rebuild the private all-rap corpus.")
    add_build_arguments(corpus_parser)
    add_profile_arguments(corpus_parser)

    source_plan_parser = subparsers.add_parser(
        "plan-corpus", help="Plan source partitions and token requirements for larger scratch models."
    )
    add_source_planning_arguments(source_plan_parser)

    gutenberg_parser = subparsers.add_parser(
        "acquire-gutenberg", help="Run a bounded Project Gutenberg acquisition pilot."
    )
    add_gutenberg_arguments(gutenberg_parser)

    gutenberg_review_parser = subparsers.add_parser(
        "review-gutenberg", help="Review acquired Gutenberg records for item-level rights and quality evidence."
    )
    add_gutenberg_review_arguments(gutenberg_review_parser)

    inspect_parser = subparsers.add_parser(
        "inspect-corpus", help="Generate a composition and governance report for a scratch corpus."
    )
    add_inspect_arguments(inspect_parser)

    ingest_parser = subparsers.add_parser(
        "ingest-source", help="Create an ingest manifest for a source registry entry."
    )
    add_ingest_arguments(ingest_parser)

    audit_parser = subparsers.add_parser(
        "audit-source", help="Run source admission gates against a source registry entry."
    )
    add_audit_arguments(audit_parser)

    admit_parser = subparsers.add_parser(
        "admit-source", help="Write a training-admission manifest for a source."
    )
    add_admit_arguments(admit_parser)

    revoke_parser = subparsers.add_parser(
        "revoke-source", help="Write a non-destructive revocation/tombstone plan for a source."
    )
    add_revoke_arguments(revoke_parser)

    migrate_parser = subparsers.add_parser(
        "migrate-source", help="Create a deterministic sidecar governance index for a legacy source."
    )
    add_migrate_arguments(migrate_parser)

    verify_profile_parser = subparsers.add_parser(
        "verify-profile", help="Verify that a profile manifest contains only admitted, non-revoked sources."
    )
    add_verify_profile_arguments(verify_profile_parser)

    impact_parser = subparsers.add_parser(
        "revocation-impact", help="Report profile, shard, tokenizer, and lineage impact for a source."
    )
    add_revocation_impact_arguments(impact_parser)

    materialize_parser = subparsers.add_parser(
        "materialize-profile", help="Write trainable JSONL splits from admitted profile sources."
    )
    add_materialize_profile_arguments(materialize_parser)

    tokenizer_parser = subparsers.add_parser("train-tokenizer", help="Train BPE and create fixed token shards.")
    add_tokenizer_arguments(tokenizer_parser)

    pretrain_parser = subparsers.add_parser("pretrain", help="Pretrain the random-initialized 30M-class model.")
    add_training_arguments(pretrain_parser, mode="base")

    sft_parser = subparsers.add_parser("sft", help="Full-finetune a scratch checkpoint on structured lyrics.")
    add_training_arguments(sft_parser, mode="sft")

    evaluation_parser = subparsers.add_parser("evaluate", help="Generate and evaluate scratch checkpoints.")
    add_evaluation_arguments(evaluation_parser)

    compliance_parser = subparsers.add_parser(
        "analyze-compliance", help="Classify exact-line failures in saved scratch generations."
    )
    add_compliance_arguments(compliance_parser)

    release_parser = subparsers.add_parser(
        "freeze-checkpoint", help="Create a loadable, hashed, versioned checkpoint release."
    )
    add_release_arguments(release_parser)

    sweep_parser = subparsers.add_parser(
        "sweep-decoding", help="Run the bounded scratch decoding and line-control sweep."
    )
    add_sweep_arguments(sweep_parser)

    router_parser = subparsers.add_parser(
        "generate-controlled", help="Generate through the versioned scratch inference router."
    )
    add_router_arguments(router_parser)

    comparison_parser = subparsers.add_parser(
        "build-three-way-comparison", help="Freeze systems and build the 64-prompt comparison matrix."
    )
    add_build_comparison_arguments(comparison_parser)

    scratch_comparison_parser = subparsers.add_parser(
        "generate-three-way-scratch", help="Generate native and routed scratch comparison outputs."
    )
    add_scratch_comparison_arguments(scratch_comparison_parser)

    blind_parser = subparsers.add_parser(
        "build-three-way-packet", help="Combine three systems into an automated candidate packet and uplift report."
    )
    add_blind_comparison_arguments(blind_parser)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "build-corpus":
        result = build_profile(args) if args.profile else build_corpus(args)
    elif args.command == "plan-corpus":
        result = plan_corpus(args)
    elif args.command == "acquire-gutenberg":
        result = acquire_gutenberg(args)
    elif args.command == "review-gutenberg":
        result = review_gutenberg(args)
    elif args.command == "inspect-corpus":
        result = inspect_corpus(args)
    elif args.command == "ingest-source":
        result = ingest_source(args)
    elif args.command == "audit-source":
        result = audit_source(args)
    elif args.command == "admit-source":
        result = admit_source(args)
    elif args.command == "revoke-source":
        result = revoke_source(args)
    elif args.command == "migrate-source":
        result = migrate_source(args)
    elif args.command == "verify-profile":
        result = verify_profile(args)
    elif args.command == "revocation-impact":
        result = revocation_impact(args)
    elif args.command == "materialize-profile":
        result = materialize_profile(args)
    elif args.command == "train-tokenizer":
        result = train_and_tokenize(args)
    elif args.command == "pretrain":
        result = train(args, mode="base")
    elif args.command == "sft":
        result = train(args, mode="sft")
    elif args.command == "evaluate":
        result = evaluate(args)
    elif args.command == "analyze-compliance":
        result = analyze_compliance(args)
    elif args.command == "freeze-checkpoint":
        result = freeze_checkpoint(args)
    elif args.command == "sweep-decoding":
        result = run_sweep(args)
    elif args.command == "generate-controlled":
        result = generate_controlled(args)
    elif args.command == "build-three-way-comparison":
        result = build_comparison(args)
    elif args.command == "generate-three-way-scratch":
        result = generate_scratch_comparison(args)
    elif args.command == "build-three-way-packet":
        result = build_blind_comparison(args)
    else:  # pragma: no cover
        parser.error(f"Unknown command: {args.command}")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
