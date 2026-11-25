"""Build the AgenticPPL release files from the research repo (token-ppl). Writes files only; uploads nothing.

  python scripts/build_dataset.py --src /path/to/token-ppl --out release/ [--work DIR] [--no-verify]

Inputs (no scored model's data, no held-out model name or score):
  src/render_tb.py + data/{tb,tb2}/probe_tb.jsonl   the normalized messages, dev (tb) and held-out (tb2)
  data/{tb,tb2}/rendered/flint.jsonl          the canonical rendering (Flint's template), order and kept prefix
  data/tb/raw/sources.json                    source repos and revisions
  pumpkin/tokppl2/store/real/picks/<sha>/     the 3 frozen problem-v2 programs and their dev picks (picked.npz)
  pumpkin/tokppl2/store/real/{feats,heldout}/ the evaluator (Flint) features on the dev and held-out trajectories
  pumpkin/tokppl/data/scores.json (real.dev) + configs/panel_tb.tsv   dev TB2.1 scores by HF repo
Outputs in --out: trajectories.jsonl.gz, picks/<name>.json, scores_dev.json, MANIFEST.json (formats: agenticppl/data.py).
--verify (default): re-renders every trajectory with the toolkit's renderer and Flint's tokenizer (HF cache) and
requires the canonical text and spans byte for byte.
"""
import argparse, gzip, hashlib, json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from agenticppl.measure import nonmarkup  # noqa: E402

LICENCE = "Apache-2.0"   # both source datasets' HF cards, checked 2026-11-24
SOURCES = {   # probe `source` -> (sources.json key, HF dataset)
    "terminal-lego-traj-8k": ("lego", "Lego-X/Terminal-Lego-Traj-8k"),
    "terminal-wrench": ("wrench", "few-sh/terminal-wrench"),
}
EVALUATOR = "RecursiveSI/flint"
DEFAULT_PICKS = "obsgain-easy"   # the owner's choice (2026-11-24), with its program's own combine
PICKS = [   # name, program SHA-256, search run, combine, what it picks
    ("obsgain-nothought", "466edc356a4ec16c460c1e248a67924a2215fe95b2149d2121ee6fe68ef249f6", "20261124-184209-f2f139bd", "prob_mean",
     "Command (action) tokens whose Flint loss drops by more than 0.5 bits when the earlier terminal output is visible, "
     "with the agent's thoughts removed on both sides (flint_no_obs_no_thought - flint_no_thought > 0.5)."),
    ("obsgain-easy", "1152f4c883a9a4c3c107659676dcee32a8e4346f007604061e594d89e229dc69", "20261124-184139-05dc517c", "capped_bits",
     "Command tokens of turns >= 1 whose Flint loss drops by more than 0.5 bits when the terminal output is visible "
     "(flint_noobs - flint > 0.5), and that Flint itself predicts well (flint < 4 bits)."),
    ("obsgain-taskgain", "b7f43f6ce3d0a8c3abf44e69372ef31f61788d086bb0e8f045c1a5200bd2bbba", "20261124-184030-c684eadc", "prob_mean",
     "Command tokens of turns >= 1 whose Flint loss drops by more than 0.5 bits when the terminal output is visible; in a "
     "trajectory with none, turn-0 command tokens where the task description helps Flint by more than 1 bit "
     "(flint_no_task_desc - flint > 1, flint > 0.2)."),
]


def doc_id(source, teacher, msgs):
    """Content hash of a trajectory: SHA-256 of [source, teacher, [[role, content], ...]], first 16 hex digits. The
    source and teacher are in it because the Terminal-Lego GLM-5 and DeepSeek-V3.2 files hold 42 identical
    trajectories (same task, same messages); the dev measurement keeps both copies."""
    s = json.dumps([source, teacher, [[m["role"], m["content"]] for m in msgs]], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def trajectories(src, split, verify):
    """The canonical records of one split: dev = data/tb (the probe), heldout = data/tb2 (the new trajectories)."""
    root = {"dev": "data/tb", "heldout": "data/tb2"}[split]
    if "render_tb" not in sys.modules:
        os.environ["TB_DATA"] = "data/tb"
        sys.path.insert(0, os.path.join(src, "src"))
    import render_tb as RT
    import render_s3 as R3
    RT.PROBE, RT._ROWS = os.path.join(src, root, "probe_tb.jsonl"), None   # rows() of this split's probe file
    by_id = {r["id"]: r for r in RT.rows()}
    canon = [json.loads(l) for l in open(os.path.join(src, root, "rendered/flint.jsonl"))]
    srcs = json.load(open(os.path.join(src, "data/tb/raw/sources.json")))
    rd = None
    if verify:
        from agenticppl.render import Renderer
        rd = Renderer(EVALUATOR)
    out, dup = [], {}
    for c in canon:
        r = by_id[c["id"]]
        msgs = R3.prefix_through(r["norm"], c["n_turns"] - 1)
        msgs = [{"role": m["role"], "content": m["content"], "parts": [list(p) for p in m["parts"]]} for m in msgs]
        if rd is not None:
            text, spans = rd(msgs)
            if text != c["text"] or spans != c["spans"]:
                raise SystemExit(f"{split} {c['id']}: the toolkit's Flint rendering differs from the canonical one")
        key, repo = SOURCES[c["source"]]
        key = key + "_heldout" if split == "heldout" and key + "_heldout" in srcs else key
        i = doc_id(repo, c["teacher"], msgs)
        dup.setdefault(doc_id(repo, "", msgs), []).append(c["id"])
        out.append({"id": i, "split": split, "source": repo, "source_revision": srcs[key]["revision"],
                    "source_ref": c["id"], "licence": LICENCE, "task_id": c["task_id"], "teacher": c["teacher"],
                    "n_turns": c["n_turns"], "messages": msgs, "text": c["text"], "spans": c["spans"]})
    dups = [v for v in dup.values() if len(v) > 1]
    print(f"{split}: {len(out)} trajectories; {len(dups)} groups with the same messages under different teachers", file=sys.stderr)
    return out, dups


def run_pick(src, sha, base, feats, keymap, work):
    """The frozen program's pick on the evaluator features of a base dir, through pick_eval.pick (the harness's own
    runner, a clean subprocess, seed 0). keymap: dev feature key -> key in `feats`."""
    sys.path.insert(0, os.path.join(src, "pumpkin/tokppl2/lib"))
    import features as F
    import pick_eval as PE
    prog = os.path.join(src, "pumpkin/tokppl2/store/real/picks", sha, "program")
    if F.tree_sha(prog) != sha:
        raise SystemExit(f"program {sha[:12]} does not match its SHA-256")
    names = json.load(open(os.path.join(prog, "features.json")))
    os.makedirs(work, exist_ok=True)
    return PE.pick(prog, PE.FeatureSet(base, feats, {n: keymap(k) for n, k in names.items()}), work, seed=0)


def picks(src, docs_by_split, work):
    """Each pick set's ranges on every split, {doc id: [[start, end], ...]} in canonical bytes; every range inside one
    action span. dev: the stored dev picks (picked.npz), checked against a fresh run of pick; heldout: pick run on
    the held-out evaluator features (store/real/heldout: the cached transforms of the dev features)."""
    S = os.path.join(src, "pumpkin/tokppl2/store/real")
    bases = {"dev": os.path.join(S, "feats/_base"), "heldout": os.path.join(S, "heldout/base")}
    for sp, docs in docs_by_split.items():
        base = [json.loads(l) for l in open(os.path.join(bases[sp], "flint.jsonl"))]
        assert len(base) == len(docs) and all(b["text"] == d["text"] for b, d in zip(base, docs)), \
            f"{sp}: the picks' base docs are not the canonical docs in order"
    hkey = {}
    for k in os.listdir(os.path.join(S, "heldout/feats")):
        mp = os.path.join(S, "heldout/feats", k, "manifest.json")
        if os.path.exists(os.path.join(S, "heldout/feats", k, "DONE")):
            m = json.load(open(mp))
            hkey[(m["evaluator"], m["transform_sha"])] = k

    def heldout_key(dev_key):
        m = json.load(open(os.path.join(S, "feats", dev_key, "manifest.json")))
        return hkey[(m["evaluator"], m["transform_sha"])]
    sets = []
    for name, sha, run, combine, desc in PICKS:
        ranges, counts = {}, {}
        for sp, docs in docs_by_split.items():
            if sp == "dev":
                z = dict(np.load(os.path.join(S, "picks", sha, "picked.npz")))
                fresh = run_pick(src, sha, bases["dev"], os.path.join(S, "feats"), lambda k: k, os.path.join(work, name, "dev"))
                srt = np.lexsort((z["start"], z["doc"]))
                assert all(np.array_equal(np.asarray(z[c])[srt], np.asarray(fresh[c])) for c in ("doc", "start", "end", "weight")), \
                    f"{name}: pick run again on the dev features differs from the stored dev picks"
            else:
                z = run_pick(src, sha, bases["heldout"], os.path.join(S, "heldout/feats"), heldout_key, os.path.join(work, name, sp))
            nb = 0
            for c, a, b in zip(z["doc"].tolist(), z["start"].tolist(), z["end"].tolist()):
                d = docs[c]
                spn = [x for x in nonmarkup(d["spans"]) if x[0] <= a and b <= x[1]]
                assert len(spn) == 1 and spn[0][2] == "action" and b > a, (name, sp, c, a, b)
                ranges.setdefault(d["id"], []).append([a, b])
                nb += b - a
            counts[sp] = {"ranges": int(len(z["doc"])), "bytes": int(nb), "docs": int(len(np.unique(z["doc"])))}
            print(f"{name} {sp}: {counts[sp]}", file=sys.stderr)
        for v in ranges.values():
            v.sort()
        sets.append({"name": name, "default": name == DEFAULT_PICKS, "combine": combine, "splits": list(docs_by_split),
                     "evaluator": EVALUATOR, "program_sha256": sha, "search_run": run, "description": desc,
                     "coordinates": "byte ranges [start, end) of the trajectory's canonical text",
                     "counts": counts, "ranges": ranges})
    return sets


def dev_scores(src):
    rows = [l.rstrip("\n").split("\t") for l in open(os.path.join(src, "configs/panel_tb.tsv")) if l.strip() and not l.startswith("#")]
    hdr = rows[0]
    repo = {r[0]: dict(zip(hdr, r))["hf_repo"] for r in rows[1:]}
    dev = json.load(open(os.path.join(src, "pumpkin/tokppl/data/scores.json")))["real"]["dev"]
    return {"benchmark": "Terminal-Bench 2.1 (Artificial Analysis; Terminus 2, 89 tasks, 3 runs each)",
            "keys": {"R": "the model's Terminal-Bench 2.1 score: reasoning mode at the highest effort, or AA's single score for a model with one mode"},
            "split": "dev", "models": {repo[s]: {"R": v.get("R")} for s, v in sorted(dev.items())}}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", required=True, help="the token-ppl repo")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(HERE), "release"))
    ap.add_argument("--work", default=os.environ.get("TMPDIR", "/tmp"), help="scratch dir for the pick runs")
    ap.add_argument("--no-verify", action="store_true", help="skip the re-rendering check (no Flint tokenizer needed)")
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "picks"), exist_ok=True)
    by_split, dups = {}, {}
    for sp in ("dev", "heldout"):
        by_split[sp], dups[sp] = trajectories(a.src, sp, not a.no_verify)
    ids = [d["id"] for v in by_split.values() for d in v]
    assert len(ids) == len(set(ids)), "content hash collision"
    files = {}
    p = os.path.join(a.out, "trajectories.jsonl.gz")
    with open(p, "wb") as raw, gzip.GzipFile(filename="", fileobj=raw, mode="wb", compresslevel=9, mtime=0) as gz:   # same input, same bytes
        for v in by_split.values():
            for d in v:
                gz.write((json.dumps(d, ensure_ascii=False) + "\n").encode())
    files["trajectories.jsonl.gz"] = p
    sets = picks(a.src, by_split, os.path.join(a.work, "agenticppl-picks"))
    for s in sets:
        p = os.path.join(a.out, "picks", s["name"] + ".json")
        json.dump(s, open(p, "w"), separators=(",", ":"))
        files[f"picks/{s['name']}.json"] = p
    p = os.path.join(a.out, "scores_dev.json")
    json.dump(dev_scores(a.src), open(p, "w"), indent=1)
    files["scores_dev.json"] = p
    src_counts = {}
    for v in by_split.values():
        for d in v:
            src_counts.setdefault(d["source"], {}).setdefault(d["split"], 0)
            src_counts[d["source"]][d["split"]] += 1
    man = {"name": "AgenticPPL", "splits": {sp: len(v) for sp, v in by_split.items()},
           "default_picks": DEFAULT_PICKS,
           "pick_sets": {s["name"]: {"combine": s["combine"], "counts": s["counts"]} for s in sets},
           "sources": {k: {"trajectories": v, "licence": LICENCE} for k, v in sorted(src_counts.items())},
           "evaluator": EVALUATOR, "scores": "dev models only (Artificial Analysis); held-out model scores are not released",
           "same_messages_different_teacher": dups,
           "files": {k: {"bytes": os.path.getsize(v), "sha256": sha256(v)} for k, v in files.items()}}
    json.dump(man, open(os.path.join(a.out, "MANIFEST.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in man.items() if k != "same_messages_different_teacher"}, indent=1))


if __name__ == "__main__":
    main()
