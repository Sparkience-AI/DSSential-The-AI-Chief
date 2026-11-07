"""Bounded, immutable-by-contract ingestion services."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO, StringIO

import pandas as pd

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_ROWS = 250_000
MAX_COLUMNS = 250


@dataclass(frozen=True)
class DatasetSnapshot:
    """Input identity plus a private source frame.

    Callers receive a deep copy through ``frame`` so transformations cannot
    silently mutate the uploaded source used by later analyses.
    """

    name: str
    sha256: str
    rows: int
    columns: int
    _source: pd.DataFrame

    @property
    def frame(self) -> pd.DataFrame:
        return self._source.copy(deep=True)


def load_csv(payload: bytes, *, name: str = "dataset.csv") -> DatasetSnapshot:
    if not payload:
        raise ValueError("The uploaded CSV is empty.")
    if len(payload) > MAX_UPLOAD_BYTES:
        raise ValueError(f"CSV exceeds the {MAX_UPLOAD_BYTES // 1_048_576} MiB limit.")
    try:
        header = next(csv.reader(StringIO(payload.decode("utf-8-sig"))))
    except (UnicodeDecodeError, StopIteration, csv.Error) as exc:
        raise ValueError("The CSV header is invalid or is not UTF-8 encoded.") from exc
    if len(header) != len(set(header)):
        raise ValueError("CSV column names must be unique.")
    frame = pd.read_csv(BytesIO(payload))
    if frame.empty:
        raise ValueError("The CSV has no data rows.")
    if len(frame) > MAX_ROWS or len(frame.columns) > MAX_COLUMNS:
        raise ValueError(f"CSV exceeds the {MAX_ROWS:,}-row or {MAX_COLUMNS}-column limit.")
    return DatasetSnapshot(
        name=name,
        sha256=sha256(payload).hexdigest(),
        rows=len(frame),
        columns=len(frame.columns),
        _source=frame.copy(deep=True),
    )
