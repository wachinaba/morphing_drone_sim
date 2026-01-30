from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CsvLogger:
    """
    毎ステップCSVを書き出す最小ロガー。
    - 依存追加なし
    - flush頻度を選べる（MVPは毎行flushでもOK）
    """

    path: Path
    fieldnames: list[str]
    flush_every: int = 1

    _fp = None
    _writer: csv.DictWriter | None = None
    _n: int = 0

    def open(self):
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fp = self.path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._fp, fieldnames=list(self.fieldnames), extrasaction="ignore")
        self._writer.writeheader()
        self._fp.flush()

    def write(self, row: dict):
        if self._writer is None or self._fp is None:
            raise RuntimeError("CsvLogger is not open. Call open() first.")
        self._writer.writerow(row)
        self._n += 1
        if int(self.flush_every) > 0 and (self._n % int(self.flush_every) == 0):
            self._fp.flush()

    def close(self):
        if self._fp is not None:
            try:
                self._fp.flush()
            except Exception:
                pass
            try:
                self._fp.close()
            finally:
                self._fp = None
                self._writer = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False





