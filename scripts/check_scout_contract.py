#!/usr/bin/env python3
"""
Validate Scout topic contract sync between backend state and frontend types.
"""

from __future__ import annotations

import re
from pathlib import Path


def _extract_backend_keys(path: Path) -> set[str]:
    text = path.read_text(encoding="utf-8")
    m = re.search(r"def to_dict\(self\) -> dict:\n\s+return \{(.*?)\n\s+\}", text, re.S)
    if not m:
        raise RuntimeError("Cannot parse backend ScoutTopic.to_dict block")
    block = m.group(1)
    return set(re.findall(r'"([a-zA-Z_][a-zA-Z0-9_]*)"\s*:', block))


def _extract_frontend_keys(path: Path) -> set[str]:
    text = path.read_text(encoding="utf-8")
    m = re.search(r"export interface ScoutTopic \{(.*?)\n\}", text, re.S)
    if not m:
        raise RuntimeError("Cannot parse frontend ScoutTopic interface")
    block = m.group(1)
    return set(re.findall(r"\n\s+([a-zA-Z_][a-zA-Z0-9_]*)\??:", block))


def main() -> int:
    backend = _extract_backend_keys(Path("api/state.py"))
    frontend = _extract_frontend_keys(Path("frontend/lib/api.ts"))

    missing_in_front = sorted(backend - frontend)
    extra_in_front = sorted(frontend - backend)
    if missing_in_front:
        print("❌ ScoutTopic fields missing in frontend/lib/api.ts:")
        for k in missing_in_front:
            print(f"  - {k}")
        return 1

    # Extras are allowed only for frontend-specific helper fields? For ScoutTopic we keep strict sync.
    if extra_in_front:
        print("❌ Extra ScoutTopic fields in frontend/lib/api.ts not present in backend:")
        for k in extra_in_front:
            print(f"  - {k}")
        return 1

    print("✅ Scout topic contract synced: backend -> frontend")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
