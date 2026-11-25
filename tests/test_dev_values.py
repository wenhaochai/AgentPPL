"""Checks the toolkit against the research harness on dev models, from their per-token dumps (CPU, no GPU).

For each model and split: the toolkit renders every release trajectory of the split with the model's chat template
and tokenizes it; the text and spans must equal the harness's rendering and the token byte offsets must equal the
dump's. Then the toolkit maps each pick set onto the model's tokens and combines the dump's bits. Each value must
equal the harness's to 1e-9 (plain PPL to 1e-6):
  split dev (1,401 trajectories; results/tb_top20full, results/v2_dev/best3_values.json)
    obsgain-easy (1152f4c8, capped_bits)     = best3 [1]       obsgain-nothought (466edc35, prob_mean) = best3 [0]
    each program's own combine.py on the toolkit's rows = best3 [0], [1], [2]
    plain PPL = the dump's summary bpb_all (summed in float32 there)
  split heldout (960 new trajectories; results/tb2_top20full, the pre-registered dev cell
    pumpkin/tokppl2/store/real/heldout_dev/out/values.json: dev models only)
    the same, with the programs 05dc517c, f2f139bd, c684eadc and minus plain_ppl of that file
--dumps-only: every dev model with a dump, the harness's renderings in place of the tokenizer; also prints the
Spearman rho of every pick set and of minus plain PPL with the dev scores.

  AGENTICPPL_SRC=/path/to/token-ppl python tests/test_dev_values.py [--split dev|heldout] [--models a,b] [--dumps-only]
  pytest tests/   (dev on the default models with their tokenizers, held-out from the dumps; skipped without the repo)
"""
import argparse, importlib.util, json, os, sys, time, zipfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from agenticppl.data import default_dir, load_scores, pick_sets, trajectories  # noqa: E402
from agenticppl.measure import agenticppl_of, gather, plain_ppl, read_jsonl, same_spans, spearman, token_bytes  # noqa: E402

SRC = os.environ.get("AGENTICPPL_SRC", "")   # the research repo (token-ppl) with the harness dumps; tests skip without it
DEFAULT = ["qwen3-5-4b", "gemma-4-12b", "k2-horizon-7b", "ministral-3-8b", "granite-4-2-3b"]
TOL = {"plain_ppl": 1e-6}   # relative; 1e-9 for the rest
ROOT = {"dev": "tb", "heldout": "tb2"}
RUN = {"obsgain-nothought": "f2f139bd", "obsgain-easy": "05dc517c", "obsgain-taskgain": "c684eadc"}
BEST3 = {"obsgain-nothought": 0, "obsgain-easy": 1, "obsgain-taskgain": 2}


def npz_arrays(path):
    """The arrays of an uncompressed .npz as read-only memmaps (the dumps' top-20 arrays are large)."""
    out = {}
    with zipfile.ZipFile(path) as z, open(path, "rb") as f:
        for info in z.infolist():
            name = info.filename[:-4]
            if info.compress_type != zipfile.ZIP_STORED:
                with z.open(info) as m:
                    out[name] = np.lib.format.read_array(m, allow_pickle=False)
                continue
            f.seek(info.header_offset)
            h = f.read(30)
            f.seek(info.header_offset + 30 + int.from_bytes(h[26:28], "little") + int.from_bytes(h[28:30], "little"))
            v = np.lib.format.read_magic(f)
            shape, fortran, dtype = (np.lib.format.read_array_header_1_0 if v == (1, 0) else np.lib.format.read_array_header_2_0)(f)
            if dtype.kind == "U" or int(np.prod(shape)) == 0:
                out[name] = np.fromfile(f, dtype=dtype, count=int(np.prod(shape))).reshape(shape)
            else:
                out[name] = np.memmap(path, dtype=dtype, mode="r", offset=f.tell(), shape=shape, order="F" if fortran else "C")
    return out


def load_combine(sha):
    p = os.path.join(SRC, "pumpkin/tokppl2/store/real/picks", sha, "program/combine.py")
    spec = importlib.util.spec_from_file_location("combine_" + sha[:8], p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.combine


def range_weights(docs, ps):
    """The weights the source pick gave its ranges: b7f43f6c 1 / (k + 24) for the k ranges of a doc, the others 1."""
    w = []
    for d in docs:
        k = len(ps["ranges"].get(d["id"], []))
        w += [1.0 / (k + 24.0) if ps["program_sha256"].startswith("b7f43f6c") else 1.0] * k
    return np.array(w, np.float64)


def native_value(docs, ps, model, z, rows_of):
    """The source program's own combine.py on the toolkit's rows (the harness's x arrays)."""
    doc_rows, rbytes, dropped = gather(docs, ps["ranges"], model)
    assert dropped == 0
    w = range_weights(docs, ps)
    g = np.concatenate([rows_of[i][rows] for i, rows, _ in doc_rows]).astype(np.int64)
    rid = np.concatenate([r for _, _, r in doc_rows]).astype(np.int64)
    x = {"n_ranges": np.int64(len(rbytes)), "range_weight": w, "range_bytes": rbytes.astype(np.int64), "rid": rid,
         "bits": np.asarray(z["bits"][g], np.float32), "ent_lb": np.asarray(z["ent_lb"][g], np.float32),
         "topk_lp": np.asarray(z["topk_lp"][g], np.float32),
         "nbytes": (np.asarray(z["end"][g]) - np.asarray(z["start"][g])).astype(np.int32), "weight": w[rid]}
    return float(load_combine(ps["program_sha256"])(x))


def panel_repos():
    rows = [l.rstrip("\n").split("\t") for l in open(os.path.join(SRC, "configs/panel_tb.tsv")) if l.strip() and not l.startswith("#")]
    return {r[0]: dict(zip(rows[0], r))["hf_repo"] for r in rows[1:]}


def dumps_dir(split):
    return os.path.join(SRC, f"results/{ROOT[split]}_top20full/tokens")


def expected(split):
    """slug -> {measure: the harness's value}."""
    if split == "dev":
        best3 = json.load(open(os.path.join(SRC, "results/v2_dev/best3_values.json")))
        def exp(slug):
            e = {"plain_ppl": json.load(open(os.path.join(dumps_dir(split), slug + ".json")))["bpb_all"]}
            for n, i in BEST3.items():
                e[n + ":native"] = best3[i]["values"].get(slug)
            e["obsgain-easy"], e["obsgain-nothought"] = e["obsgain-easy:native"], e["obsgain-nothought:native"]
            return e
        return exp
    v = json.load(open(os.path.join(SRC, "pumpkin/tokppl2/store/real/heldout_dev/out/values.json")))
    assert v["cell"] == "dev"
    def exp(slug):
        e = {"plain_ppl": -v["plain_ppl"][slug]}
        for n, run in RUN.items():
            e[n + ":native"] = v["programs"][run]["values"].get(slug)
        e["obsgain-easy"], e["obsgain-nothought"] = e["obsgain-easy:native"], e["obsgain-nothought:native"]
        return e
    return exp


def check_model(slug, repo, split, docs, sets, exp, use_tokenizer):
    """Toolkit values of one model on one split from its dump; returns (values, failures)."""
    t0 = time.time()
    z = npz_arrays(os.path.join(dumps_dir(split), slug + ".npz"))
    hdocs = {d["id"]: d for d in read_jsonl(os.path.join(SRC, f"data/{ROOT[split]}/rendered", slug + ".jsonl"))}
    zid = [str(x) for x in z["doc_ids"]]
    zb = np.searchsorted(np.asarray(z["doc"]), np.arange(len(zid) + 1))
    zpos = {d: k for k, d in enumerate(zid)}
    zs, ze, zbits = np.asarray(z["start"], np.int64), np.asarray(z["end"], np.int64), np.asarray(z["bits"], np.float64)
    fails, model, bits, rows_of = [], {}, {}, {}
    rd = None
    if use_tokenizer:
        from agenticppl.render import Renderer
        rd = Renderer(repo)
    for d in docs:
        h = hdocs[d["source_ref"]]
        k = zpos[d["source_ref"]]
        lo, hi = int(zb[k]), int(zb[k + 1])
        if rd is not None:
            text, spans = rd(d["messages"])
            if text != h["text"] or spans != h["spans"]:
                fails.append(f"{d['source_ref']}: rendering differs from the harness")
                continue
            enc = rd.tok(text, add_special_tokens=False, return_offsets_mapping=True)
            s, e = token_bytes(text, enc["offset_mapping"])
            if not (np.array_equal(s, zs[lo:hi]) and np.array_equal(e, ze[lo:hi])):
                fails.append(f"{d['source_ref']}: token offsets differ from the dump")
                continue
        else:
            text, spans = h["text"], h["spans"]
        ok = same_spans(d["text"], d["spans"], text, spans)
        if not ok:
            fails.append(f"{d['source_ref']}: non-markup spans differ from the canonical rendering")
        model[d["id"]] = (spans if ok else None, zs[lo:hi], ze[lo:hi])
        bits[d["id"]] = zbits[lo:hi]
        rows_of[d["id"]] = np.arange(lo, hi)
    vals = {"plain_ppl": plain_ppl(docs, model, bits)}
    for name, ps in sets.items():
        vals[name] = agenticppl_of(docs, ps["ranges"], model, bits, ps["combine"])["agenticppl"]
        vals[name + ":native"] = native_value(docs, ps, model, z, rows_of)
    for key, want in exp(slug).items():
        if want is None or key not in vals:
            continue
        if not abs(vals[key] - want) <= TOL.get(key, 1e-9) * max(1.0, abs(want)):
            fails.append(f"{key}: toolkit {vals[key]!r} harness {want!r}")
    print(f"{split} {slug}: {'OK' if not fails else 'FAIL'} ({time.time() - t0:.0f} s) " +
          " ".join(f"{k}={v:.6f}" for k, v in vals.items()), flush=True)
    for f in fails[:10]:
        print("   ", f)
    return vals, fails


def run(models, split, use_tokenizer):
    data = default_dir()
    docs = [d for d in trajectories(data) if d["split"] == split]
    sets = pick_sets(data)
    exp, repos = expected(split), panel_repos()
    allv, nfail = {}, 0
    for m in models:
        v, f = check_model(m, repos[m], split, docs, sets, exp, use_tokenizer)
        allv[m], nfail = v, nfail + len(f)
    return allv, nfail, repos, data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=["dev", "heldout"])
    ap.add_argument("--models", default=",".join(DEFAULT))
    ap.add_argument("--dumps-only", action="store_true", help="every dev model with a dump; harness renderings, no tokenizer")
    a = ap.parse_args()
    dev_scores = load_scores(os.path.join(default_dir(), "scores_dev.json"))
    repos = panel_repos()
    if a.dumps_only:   # dev models only: the scores file holds exactly them
        models = sorted(f[:-4] for f in os.listdir(dumps_dir(a.split)) if f.endswith(".npz") and repos.get(f[:-4]) in dev_scores)
    else:
        models = [m for m in a.models.split(",") if m]
    allv, nfail, repos, data = run(models, a.split, not a.dumps_only)
    if a.dumps_only:
        print(f"\n{a.split} trajectories x dev models: Spearman rho (n)\tR\tN")
        for k in ["plain_ppl"] + [k for k in next(iter(allv.values())) if k != "plain_ppl"]:
            sign = -1.0 if k == "plain_ppl" else 1.0
            cells = []
            for t in ("R", "N"):
                ms = [m for m in models if dev_scores[repos[m]].get(t) is not None]
                cells.append(f"{spearman([sign * allv[m][k] for m in ms], [dev_scores[repos[m]][t] for m in ms]):+.4f} ({len(ms)})")
            print(("-" if sign < 0 else "") + k, *cells, sep="\t")
    print(f"\n{a.split}: {len(models)} models, {nfail} failures")
    sys.exit(1 if nfail else 0)


def test_dev_values():
    import pytest
    if not os.path.isdir(dumps_dir("dev")):
        pytest.skip("research repo with the dev dumps not found (AGENTICPPL_SRC)")
    assert run(DEFAULT, "dev", True)[1] == 0


def test_heldout_values():
    import pytest
    if not os.path.isdir(dumps_dir("heldout")):
        pytest.skip("research repo with the held-out dumps of the dev models not found (AGENTICPPL_SRC)")
    assert run(DEFAULT, "heldout", False)[1] == 0


if __name__ == "__main__":
    main()
