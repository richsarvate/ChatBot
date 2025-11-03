from __future__ import annotations

import argparse

from app.ingest import ingest_emails


def main() -> None:
    parser = argparse.ArgumentParser(description="Utilities for the email QA prototype.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Parse emails and update the index.")
    ingest_parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Drop the existing Chroma index before ingesting.",
    )
    ingest_parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume ingestion from last checkpoint (after API limit reached).",
    )
    ingest_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of emails to process (for testing).",
    )

    args = parser.parse_args()

    if args.command == "ingest":
        stats = ingest_emails(rebuild=args.rebuild, resume=args.resume, limit=args.limit)
        print(
            f"Processed {stats.processed_messages} emails and {stats.processed_chunks} chunks."
        )


if __name__ == "__main__":
    main()
