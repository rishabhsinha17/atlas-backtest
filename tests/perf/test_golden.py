"""Bit-for-bit golden results (docs/PERF.md): every case in golden.CASES, run on the real cache with the
current code, must reproduce the digests recorded at the pre-optimisation commit (f9702e1) exactly:
equity, returns, fills, trades, positions, ledger, fidelity scorecard (tiers + reasons), status/reason,
every diagnostics entry and meta (minus wall-clock fields)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import golden  # noqa: E402


@pytest.mark.slow
@pytest.mark.parametrize("case", sorted(golden.CASES))
def test_golden_case_identical(case):
    gfile = golden.GOLDEN_DIR / case / "digest.json"
    if not gfile.exists():
        pytest.skip(f"no golden recorded for {case}")
    res, secs = golden.run_case(case)
    bad = golden.compare(golden.load_golden(case), golden.digest_result(res))
    assert not bad, f"{case}: outputs differ from the golden run in {bad}"
