from __future__ import annotations


def _load_row_ids_file(path: str) -> set[int]:
    ids: set[int] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            ids.add(int(s))
    return ids
