"""Compares `agentppl score` results of dev models with the research harness (the GPU smoke test).

  python scripts/smoke_compare.py RESULT.json ...     (AGENTPPL_SRC: the token-ppl repo)

Reference per model: plain PPL = the dump's bpb_all (results/tb_top20full/tokens/<slug>.json); obsgain-nothought =
results/v2_dev/best3_values.json [0]; obsgain-easy = combine_ablation.json "1152f4c8_arith"; obsgain-taskgain = the
toolkit on the dump (no harness file holds that combine on those picks; the test shows the toolkit on the dumps equals
the harness for the other two). Acceptance: every |diff| <= 1e-3. Exit 1 otherwise.
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "tests"))
import test_dev_values as T  # noqa: E402
from agentppl.data import default_dir, pick_sets, trajectories  # noqa: E402

TOL = 1e-3


def main():
    paths = sys.argv[1:]
    slug = {v: k for k, v in T.panel_repos().items()}
    best3, exp = T.expected()
    data = default_dir()
    docs = [d for d in trajectories(data) if d["split"] == "dev"]
    sets = pick_sets(data)
    worst, bad, rows = 0.0, 0, []
    for p in paths:
        r = json.load(open(p))
        s = slug[r["model"]]
        dump, _ = T.check_model(s, r["model"], docs, sets, exp, False)
        e = exp(s)
        ref = {"plain_ppl": e["plain_ppl"], "obsgain-nothought": e["obsgain-nothought"], "obsgain-easy": e["obsgain-easy"],
               "obsgain-taskgain": dump["obsgain-taskgain"]}
        got = {"plain_ppl": r["splits"]["dev"]["plain_ppl"]} | {k: v["agentppl"] for k, v in r["splits"]["dev"]["picks"].items()}
        for k, want in ref.items():
            d = abs(got[k] - want)
            worst = max(worst, d)
            bad += d > TOL
            rows.append(f"{r['model']}\t{k}\t{got[k]:.6f}\t{want:.6f}\t{got[k] - want:+.2e}\t{'ok' if d <= TOL else 'FAIL'}")
        rows.append(f"{r['model']}\twarnings\tspecial={r['docs_with_special_tokens_in_content']} unmatched={r['docs_unmatched']}")
    print("\nmodel\tmeasure\tGPU run\treference\tdiff\t<=1e-3")
    print("\n".join(rows))
    print(f"\n{len(paths)} models, max |diff| {worst:.2e}, {bad} over {TOL}")
    sys.exit(1 if bad or not paths else 0)


if __name__ == "__main__":
    main()
