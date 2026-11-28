"""The capped_bits sign (positive bits from 0.2.0) and report's flip of older result files (CPU, no data)."""
import copy, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agenticppl import __version__  # noqa: E402
from agenticppl.cli import report, upgrade  # noqa: E402
from agenticppl.measure import capped_bits  # noqa: E402


def result(version, easy, nothought, plain):
    r = {"model": "m", "splits": {"dev": {"n_docs": 1, "picks_name": "obsgain-easy", "agenticppl": easy, "n_ranges": 1,
         "plain_ppl": plain, "picks": {"obsgain-easy": {"agenticppl": easy, "combine": "capped_bits", "n_ranges": 1},
                                       "obsgain-nothought": {"agenticppl": nothought, "combine": "prob_mean", "n_ranges": 1}}}}}
    if version is not None:
        r["agenticppl_version"] = version
    return r


def test_capped_bits_positive():
    # range 0: rows 0.5 + 3.0 (capped to 2) -> 2.5 -> capped to 2; range 1: 0.25; mean 1.125
    v, n = capped_bits(np.array([0.5, 3.0, 0.25, np.nan]), np.array([0, 0, 1, 2]), 3)
    assert n == 2 and v == 1.125


def test_upgrade_flips_legacy_only():
    for old in ("0.1.0", None):
        r = upgrade(result(old, -0.7755, 0.585, 1.52))
        s = r["splits"]["dev"]
        assert s["agenticppl"] == 0.7755 and s["picks"]["obsgain-easy"]["agenticppl"] == 0.7755
        assert s["picks"]["obsgain-nothought"]["agenticppl"] == 0.585 and s["plain_ppl"] == 1.52
    new = result(__version__, 0.7755, 0.585, 1.52)
    assert upgrade(copy.deepcopy(new)) == new
    # a legacy file whose default set is prob_mean keeps its top-level value
    r = result("0.1.0", -0.7755, 0.585, 1.52)
    r["splits"]["dev"].update(picks_name="obsgain-nothought", agenticppl=0.585)
    assert upgrade(r)["splits"]["dev"]["agenticppl"] == 0.585


def test_report_mixed_files(tmp_path, capsys):
    import json
    # four models; lower bits = higher score. Two written by 0.1.0 (minus bits), two by the current version.
    rows = [("a", "0.1.0", -0.6, 0.9, 1.0, 40.0), ("b", __version__, 0.7, 0.8, 1.1, 30.0),
            ("c", "0.1.0", -0.8, 0.7, 1.2, 20.0), ("d", __version__, 0.9, 0.6, 1.3, 10.0)]
    paths, scores = [], {"models": {}}
    for m, ver, easy, noth, plain, sc in rows:
        r = result(ver, easy, noth, plain)
        r["model"] = m
        p = tmp_path / f"{m}.json"
        p.write_text(json.dumps(r))
        paths.append(str(p))
        scores["models"][m] = {"R": sc}
    sp = tmp_path / "scores.json"
    sp.write_text(json.dumps(scores))
    report(type("A", (), {"results": paths, "scores": str(sp)})())
    out = capsys.readouterr().out
    assert "lower bits should rank higher, so negative rho means the right order" in out
    rho = {l.split("\t")[0]: l.split("\t")[1] for l in out.splitlines() if "\t" in l and "(4)" in l}
    assert rho["obsgain-easy (capped_bits)"] == "-1.000 (4)"
    assert rho["obsgain-nothought (prob_mean)"] == "+1.000 (4)"
    assert rho["plain_ppl (bits/byte)"] == "-1.000 (4)"
    assert "0.6000" in out and "-0.6000" not in out
