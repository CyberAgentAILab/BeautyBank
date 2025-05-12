from __future__ import annotations

import argparse
import logging
from skit.task import Job
from rich.logging import RichHandler

logger = logging.getLogger(__name__)


def global_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["critical", "error", "warning", "info", "debug", "rich"],
        help="Logging level.",
    )


def main(args: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Python Project Starter Kit for Researchers."
    )
    global_options(parser)
    options, extra_options = parser.parse_known_args(args)
    if options.log_level == "rich":
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            datefmt="[%X]",
            handlers=[RichHandler(rich_tracebacks=True)],
        )
    else:
        logging.basicConfig(
            level=getattr(logging, options.log_level.upper()), format="%(message)s"
        )
    config = {
        "model": {
            "name": "LogisticRegression",
            "random_state": 2024,
        }
    }
    job = Job.model_validate(config)
    job.run()


# Allow the script to be run standalone (useful during development).
if __name__ == "__main__":
    main()
