"""Позволяет запускать CLI как ``python -m floodrisk ...`` (см. SRS §10.1)."""

import sys

from floodrisk.cli.__main__ import main

if __name__ == "__main__":
    sys.exit(main())
