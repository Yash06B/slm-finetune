"""Load KEY=VALUE lines from the repo's .env (gitignored) into os.environ, without overriding."""

from __future__ import annotations

import os

from .schema import REPO_ROOT


def load_dotenv(path=REPO_ROOT / ".env") -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.removeprefix("export ").partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
