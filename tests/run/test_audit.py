"""Activity audit gate: fails if any registered strategy's baseline run (results/runs/<SID>/baseline,
produced by `python -m atlasbt.run audit --live` or `run`) is status OK but ZERO_ACTIVITY without a
declared `activity_note`. Strategies without a baseline run yet are skipped (not silently passed: the
skip reason names the command to run)."""
import pytest

from atlasbt.run.audit import audit_rows
from atlasbt.run.common import DEFAULT_RESULTS
from atlasbt.run.pipeline import RunOptions
from atlasbt.run.variants import variant_ids
from atlasbt.strategies import registry as Reg

IDS = Reg.all_ids() + variant_ids()          # proxy variants (S031@proxy ...) are gated the same way
ROWS = None


def _rows():
    global ROWS
    if ROWS is None:
        ROWS = {r["id"]: r for r in audit_rows(RunOptions(results_dir=str(DEFAULT_RESULTS)), IDS)}
    return ROWS


@pytest.mark.parametrize("sid", IDS)
def test_baseline_is_not_silently_inactive(sid):
    r = _rows()[sid]
    if r["verdict"] in ("NOT_RUN", "HOLD"):
        pytest.skip(f"{sid}: {r['verdict']} ({r['detail'][:120]}); run `python -m atlasbt.run audit --live`")
    assert r["verdict"] != "OFFENDER", f"{sid} ZERO_ACTIVITY without NOT_TESTABLE or declared reason: {r['detail']}"
    assert r["verdict"] != "LEADING_GAP", f"{sid} first fill years after the window start: {r['detail']}"
