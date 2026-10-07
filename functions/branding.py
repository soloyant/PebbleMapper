"""Product brand constants: display name, distribution slug, package token.

"Clast" is a geological term, not a brand string; only the product name lives
here. Keep this module free of heavy imports so anything can import it at
load time.
"""
from __future__ import annotations

# Human-facing display name.
APP_NAME: str = "PebbleMapper"

# Kebab-case slug: repository / PyPI distribution name.
APP_SLUG: str = "pebblemapper"

# Snake-case token: cache-dir name / importable identifier.
APP_PACKAGE: str = "pebblemapper"

APP_TAGLINE: str = "Grain sizes from photographs and drone orthos"

__all__ = ["APP_NAME", "APP_SLUG", "APP_PACKAGE", "APP_TAGLINE"]
