"""Source-checkout wrapper for the installed Habitat smoke test."""
from __future__ import annotations

from navprobe.env.habitat_smoke import build_parser, check_install, main, run

__all__ = ["build_parser", "check_install", "main", "run"]


if __name__ == "__main__":
    main()
