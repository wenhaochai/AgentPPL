"""Compares `agenticppl score` results of dev models (GPU) with the same measures on the harness's dumps (the GPU
smoke test).

  python scripts/smoke_compare.py RESULT.json ...     (AGENTICPPL_SRC: the token-ppl repo)

Reference per model, split dev: plain PPL = the dump's bpb_all; each pick set = the toolkit's combine (the one the
result names) on the dump's bits, which tests/test_dev_values.py shows equal to the harness's values. Acceptance:
every |diff| <= 1e-3. Exit 1 otherwise.
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "tests"))
import test_dev_values as T  # noqa: E402
from agenticppl.cli import upgrade  # noqa: E402
from agenticppl.data import default_dir, pick_sets, trajectories  # noqa: E402

TOL = 1e-3


def main():
    paths = sys.argv[1:]
    repos = T.panel_repos()
    slug = {v: k for k, v in repos.items()}
    data = default_dir()
    docs = [d for d in trajectories(data) if d["split"] == "dev"]
    sets = pick_sets(data)
    exp = T.expected("dev", sets)
    worst, bad, rows = 0.0, 0, []
    for p in paths:
        r = upgrade(json.load(open(p)))   # files before 0.2.0 carry minus capped bits
        s = slug[r["model"]]
        got = r["splits"]["dev"]
        for name, v in got["picks"].items():
            sets[name]["combine"] = v.get("combine", "prob_mean")   # the combine this result used
        dump, _ = T.check_model(s, r["model"], "dev", docs, sets, exp, False)
        cmp = [("plain_ppl", got["plain_ppl"], exp(s)["plain_ppl"])] + \
              [(n, v["agenticppl"], dump[n]) for n, v in got["picks"].items()]
        for k, x, want in cmp:
            d = abs(x - want)
            worst = max(worst, d)
            bad += d > TOL
            rows.append(f"{r['model']}\t{k}\t{x:.6f}\t{want:.6f}\t{x - want:+.2e}\t{'ok' if d <= TOL else 'FAIL'}")
        rows.append(f"{r['model']}\twarnings\tspecial={r['docs_with_special_tokens_in_content']} unmatched={r['docs_unmatched']}")
    print("\nmodel\tmeasure\tGPU run\treference\tdiff\t<=1e-3")
    print("\n".join(rows))
    print(f"\n{len(paths)} models, max |diff| {worst:.2e}, {bad} over {TOL}")
    sys.exit(1 if bad or not paths else 0)


if __name__ == "__main__":
    main()
