"""Build the AgentPPL release files from the research repo (token-ppl). Writes files only; uploads nothing.

  python scripts/build_dataset.py --src /path/to/token-ppl --out release/ [--no-verify]

Inputs (dev only; nothing held out is opened):
  src/render_tb.py + data/tb/probe_tb.jsonl   the normalized dev messages (render_tb.rows())
  data/tb/rendered/flint.jsonl                the canonical rendering (Flint's template), order and kept prefix
  data/tb/raw/sources.json                    source repos and revisions
  pumpkin/tokppl2/store/real/picks/<sha>/picked.npz   the dev picks of the 3 best problem-v2 programs
  pumpkin/tokppl/data/scores.json (real.dev) + configs/panel_tb.tsv   dev TB2.1 scores by HF repo
Outputs in --out: trajectories.jsonl.gz, picks/<name>.json, scores_dev.json, MANIFEST.json (formats: agentppl/data.py).
--verify (default): re-renders every trajectory with the toolkit's renderer and Flint's tokenizer (HF cache) and
requires the canonical text and spans byte for byte.
"""
import argparse, gzip, hashlib, json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from agentppl.measure import nonmarkup  # noqa: E402

LICENCE = "Apache-2.0"   # both source datasets' HF cards, checked 2026-11-24
SOURCES = {   # probe `source` -> (sources.json key, HF dataset)
    "terminal-lego-traj-8k": ("lego", "Lego-X/Terminal-Lego-Traj-8k"),
    "terminal-wrench": ("wrench", "few-sh/terminal-wrench"),
}
EVALUATOR = "RecursiveSI/flint"
PICKS = [   # name, program SHA-256, search run, what it picks (dev rho with each program's own combine: FACTS.md section 2)
    ("obsgain-nothought", "466edc356a4ec16c460c1e248a67924a2215fe95b2149d2121ee6fe68ef249f6", "20261124-184209-f2f139bd",
     "Command (action) tokens whose Flint loss drops by more than 0.5 bits when the earlier terminal output is visible, "
     "with the agent's thoughts removed on both sides (flint_no_obs_no_thought - flint_no_thought > 0.5)."),
    ("obsgain-easy", "1152f4c883a9a4c3c107659676dcee32a8e4346f007604061e594d89e229dc69", "20261124-184139-05dc517c",
     "Command tokens of turns >= 1 whose Flint loss drops by more than 0.5 bits when the terminal output is visible "
     "(flint_noobs - flint > 0.5), and that Flint itself predicts well (flint < 4 bits)."),
    ("obsgain-taskgain", "b7f43f6ce3d0a8c3abf44e69372ef31f61788d086bb0e8f045c1a5200bd2bbba", "20261124-184030-c684eadc",
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


def dev_trajectories(src, verify):
    os.environ["TB_DATA"] = "data/tb"   # the dev probe; never the held-out data root
    sys.path.insert(0, os.path.join(src, "src"))
    import render_tb as RT
    import render_s3 as R3
    assert os.path.normpath(RT.DATA) == os.path.normpath(os.path.join(src, "data/tb")), RT.DATA
    by_id = {r["id"]: r for r in RT.rows()}
    canon = [json.loads(l) for l in open(os.path.join(src, "data/tb/rendered/flint.jsonl"))]
    srcs = json.load(open(os.path.join(src, "data/tb/raw/sources.json")))
    rd = None
    if verify:
        from agentppl.render import Renderer
        rd = Renderer(EVALUATOR)
    out, seen, dup = [], set(), {}
    for c in canon:
        r = by_id[c["id"]]
        msgs = R3.prefix_through(r["norm"], c["n_turns"] - 1)
        msgs = [{"role": m["role"], "content": m["content"], "parts": [list(p) for p in m["parts"]]} for m in msgs]
        if rd is not None:
            text, spans = rd(msgs)
            if text != c["text"] or spans != c["spans"]:
                raise SystemExit(f"{c['id']}: the toolkit's Flint rendering differs from the canonical one")
        key, repo = SOURCES[c["source"]]
        i = doc_id(repo, c["teacher"], msgs)
        assert i not in seen, f"content hash collision {i}"
        seen.add(i)
        dup.setdefault(doc_id(repo, "", msgs), []).append(c["id"])
        out.append({"id": i, "split": "dev", "source": repo, "source_revision": srcs[key]["revision"],
                    "source_ref": c["id"], "licence": LICENCE, "task_id": c["task_id"], "teacher": c["teacher"],
                    "n_turns": c["n_turns"], "messages": msgs, "text": c["text"], "spans": c["spans"]})
    dups = [v for v in dup.values() if len(v) > 1]
    print(f"{len(out)} trajectories; {len(dups)} groups with the same messages under different teachers", file=sys.stderr)
    return out, dups


def dev_picks(src, docs):
    """Each pick set's dev ranges, {doc id: [[start, end], ...]} in canonical bytes; every range inside one action span."""
    base = [json.loads(l) for l in open(os.path.join(src, "pumpkin/tokppl2/store/real/feats/_base/flint.jsonl"))]
    assert len(base) == len(docs) and all(b["text"] == d["text"] for b, d in zip(base, docs)), \
        "the picks' base docs are not the canonical docs in order"
    sets = []
    for name, sha, run, desc in PICKS:
        z = np.load(os.path.join(src, "pumpkin/tokppl2/store/real/picks", sha, "picked.npz"))
        ranges, nbytes = {}, 0
        for c, a, b in zip(z["doc"].tolist(), z["start"].tolist(), z["end"].tolist()):
            d = docs[c]
            sp = [x for x in nonmarkup(d["spans"]) if x[0] <= a and b <= x[1]]
            assert len(sp) == 1 and sp[0][2] == "action" and b > a, (name, c, a, b)
            ranges.setdefault(d["id"], []).append([a, b])
            nbytes += b - a
        for v in ranges.values():
            v.sort()
        sets.append({"name": name, "splits": ["dev"], "evaluator": EVALUATOR, "program_sha256": sha, "search_run": run,
                     "description": desc, "coordinates": "byte ranges [start, end) of the trajectory's canonical text",
                     "n_ranges": int(len(z["doc"])), "n_bytes": int(nbytes), "n_docs": len(ranges), "ranges": ranges})
    return sets


def dev_scores(src):
    rows = [l.rstrip("\n").split("\t") for l in open(os.path.join(src, "configs/panel_tb.tsv")) if l.strip() and not l.startswith("#")]
    hdr = rows[0]
    repo = {r[0]: dict(zip(hdr, r))["hf_repo"] for r in rows[1:]}
    dev = json.load(open(os.path.join(src, "pumpkin/tokppl/data/scores.json")))["real"]["dev"]
    return {"benchmark": "Terminal-Bench 2.1 (Artificial Analysis; Terminus 2, 89 tasks, 3 runs each)",
            "keys": {"R": "reasoning mode (highest effort)", "N": "non-reasoning mode; a model with one AA score has it as both R and N"},
            "split": "dev", "models": {repo[s]: {"R": v.get("R"), "N": v.get("N")} for s, v in sorted(dev.items())}}


def heldout(src, out):
    """HOOK (held-out release, not now): the held-out trajectories and the held-out picks are sealed until the
    held-out judgement. When the owner releases them, this adds split "heldout": the held-out trajectories in the same
    record format (canonical Flint rendering, content-hash ids), and each pick set's held-out ranges (the frozen pick
    program run on Flint's passes over the held-out trajectories), appended to picks/<name>.json with "heldout" added
    to its "splits". Nothing here may read the held-out data before then."""
    raise SystemExit("held-out trajectories and picks are sealed until the held-out judgement; not built")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", required=True, help="the token-ppl repo")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(HERE), "release"))
    ap.add_argument("--no-verify", action="store_true", help="skip the re-rendering check (no Flint tokenizer needed)")
    ap.add_argument("--heldout", action="store_true", help="add the held-out split (sealed: refuses)")
    a = ap.parse_args()
    if a.heldout:
        heldout(a.src, a.out)
    os.makedirs(os.path.join(a.out, "picks"), exist_ok=True)
    docs, dups = dev_trajectories(a.src, not a.no_verify)
    files = {}
    p = os.path.join(a.out, "trajectories.jsonl.gz")
    with open(p, "wb") as raw, gzip.GzipFile(filename="", fileobj=raw, mode="wb", compresslevel=9, mtime=0) as gz:   # same input, same bytes
        for d in docs:
            gz.write((json.dumps(d, ensure_ascii=False) + "\n").encode())
    files["trajectories.jsonl.gz"] = p
    for s in dev_picks(a.src, docs):
        p = os.path.join(a.out, "picks", s["name"] + ".json")
        json.dump(s, open(p, "w"), separators=(",", ":"))
        files[f"picks/{s['name']}.json"] = p
    p = os.path.join(a.out, "scores_dev.json")
    json.dump(dev_scores(a.src), open(p, "w"), indent=1)
    files["scores_dev.json"] = p
    by_src = {}
    for d in docs:
        by_src[d["source"]] = by_src.get(d["source"], 0) + 1
    man = {"name": "AgentPPL", "splits": {"dev": len(docs), "heldout": "not released (sealed)"},
           "sources": {k: {"trajectories": v, "licence": LICENCE} for k, v in sorted(by_src.items())},
           "evaluator": EVALUATOR,
           "same_messages_different_teacher": dups,
           "files": {k: {"bytes": os.path.getsize(v), "sha256": sha256(v)} for k, v in files.items()}}
    json.dump(man, open(os.path.join(a.out, "MANIFEST.json"), "w"), indent=1)
    print(json.dumps(man, indent=1))


if __name__ == "__main__":
    main()
