from __future__ import annotations

import pandas as pd
import pytest

from dssential.ingestion import MAX_UPLOAD_BYTES, load_csv


def test_snapshot_returns_deep_copies() -> None:
    snapshot = load_csv(b"date,value\n2025-01-01,1\n2025-01-02,2\n")
    first = snapshot.frame
    first.loc[0, "value"] = 999
    assert snapshot.frame.loc[0, "value"] == 1
    assert len(snapshot.sha256) == 64


def test_ingestion_rejects_empty_duplicate_and_oversized_inputs() -> None:
    with pytest.raises(ValueError, match="empty"):
        load_csv(b"")
    with pytest.raises(ValueError, match="unique"):
        load_csv(b"a,a\n1,2\n")
    with pytest.raises(ValueError, match="limit"):
        load_csv(b"x" * (MAX_UPLOAD_BYTES + 1))


def test_source_is_not_changed_by_copy_operations() -> None:
    snapshot = load_csv(b"x,y\n1,2\n3,4\n")
    transformed = snapshot.frame.assign(x=lambda frame: frame["x"] * 10)
    pd.testing.assert_series_equal(snapshot.frame["x"], pd.Series([1, 3], name="x"))
    assert transformed["x"].tolist() == [10, 30]
