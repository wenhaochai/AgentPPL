"""Checks the toolkit against the research harness on dev models, from their per-token dumps (CPU, no GPU).

For each model: the toolkit renders every release trajectory with the model's chat template and tokenizes it; the
text and spans must equal the harness's rendering and the token byte offsets must equal the dump's. Then the toolkit
maps each pick set onto the model's tokens and combines the dump's bits; the values must equal the harness's:
  obsgain-nothought (466edc35)   AgentPPL = results/v2_dev/best3_values.json [0] (that program's own combine)
  obsgain-easy (1152f4c8)        AgentPPL = results/v2_dev/combine_ablation.json "1152f4c8_arith"
  all 3 sets, each program's own combine.py on the toolkit's rows = best3_values.json [0], [1], [2]
  plain PPL                      = the dump's own summary bpb_all (results/tb_top20full/tokens/<slug>.json; its sum
                                 ran in float32, so to 1e-6)
--dumps-only: every dev model with a dump, the harness's renderings in place of the tokenizer (no tokenizer needed);
also prints the dev Spearman rho of every pick set with the AgentPPL combine and of minus plain PPL.

  AGENTPPL_SRC=/path/to/token-ppl python tests/test_dev_values.py [--models slug,...] [--dumps-only]
  pytest tests/   (the default models; skipped without the research repo)
"""
import argparse, importlib.util, json, os, sys, time, zipfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from agentppl.data import default_dir, load_scores, pick_sets, trajectories  # noqa: E402
from agentppl.measure import agentppl_of, gather, plain_ppl, read_jsonl, same_spans, spearman, token_bytes  # noqa: E402

SRC = os.environ.get("AGENTPPL_SRC", "/scratch/gpfs/KARTHIKN/wc9403/project/token-ppl")
DEFAULT = ["qwen3-5-4b", "gemma-4-12b", "k2-horizon-7b", "ministral-3-8b", "granite-4-2-3b"]
TOL = {"plain_ppl": 1e-6}   # relative; 1e-9 for the rest


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


def native_value(docs, ps, model, z, rows_of, sha):
    """The source program's own combine.py on the toolkit's rows (the harness's x arrays)."""
    doc_rows, rbytes, _ = gather(docs, ps["ranges"], model)
    pk = np.load(os.path.join(SRC, "pumpkin/tokppl2/store/real/picks", sha, "picked.npz"))
    srt = np.lexsort((pk["start"], pk["doc"]))
    w = np.asarray(pk["weight"], np.float64)[srt]
    assert len(w) == len(rbytes), "weights and kept ranges differ in number"
    g = np.concatenate([rows_of[i][rows] for i, rows, _ in doc_rows]).astype(np.int64)
    rid = np.concatenate([r for _, _, r in doc_rows]).astype(np.int64)
    x = {"n_ranges": np.int64(len(rbytes)), "range_weight": w, "range_bytes": rbytes.astype(np.int64), "rid": rid,
         "bits": np.asarray(z["bits"][g], np.float32), "ent_lb": np.asarray(z["ent_lb"][g], np.float32),
         "topk_lp": np.asarray(z["topk_lp"][g], np.float32),
         "nbytes": (np.asarray(z["end"][g]) - np.asarray(z["start"][g])).astype(np.int32), "weight": w[rid]}
    return float(load_combine(sha)(x))


def panel_repos():
    rows = [l.rstrip("\n").split("\t") for l in open(os.path.join(SRC, "configs/panel_tb.tsv")) if l.strip() and not l.startswith("#")]
    return {r[0]: dict(zip(rows[0], r))["hf_repo"] for r in rows[1:]}


def check_model(slug, repo, docs, sets, exp, use_tokenizer):
    """Toolkit values of one model from its dump; returns (values, failures)."""
    t0 = time.time()
    z = npz_arrays(os.path.join(SRC, "results/tb_top20full/tokens", slug + ".npz"))
    hdocs = {d["id"]: d for d in read_jsonl(os.path.join(SRC, "data/tb/rendered", slug + ".jsonl"))}
    zid = [str(x) for x in z["doc_ids"]]
    zdoc = np.asarray(z["doc"])
    zb = np.searchsorted(zdoc, np.arange(len(zid) + 1))
    zpos = {d: k for k, d in enumerate(zid)}
    zs, ze, zbits = np.asarray(z["start"], np.int64), np.asarray(z["end"], np.int64), np.asarray(z["bits"], np.float64)
    fails, model, bits, rows_of = [], {}, {}, {}
    rd = None
    if use_tokenizer:
        from agentppl.render import Renderer
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
        vals[name] = agentppl_of(docs, ps["ranges"], model, bits)["agentppl"]
        vals[name + ":native"] = native_value(docs, ps, model, z, rows_of, ps["program_sha256"])
    for key, want in exp(slug).items():
        if want is None or key not in vals:
            continue
        if not abs(vals[key] - want) <= TOL.get(key, 1e-9) * max(1.0, abs(want)):
            fails.append(f"{key}: toolkit {vals[key]!r} harness {want!r}")
    print(f"{slug}: {'OK' if not fails else 'FAIL'} ({time.time() - t0:.0f} s) " +
          " ".join(f"{k}={v:.6f}" for k, v in vals.items()), flush=True)
    for f in fails[:10]:
        print("   ", f)
    return vals, fails


def expected():
    best3 = json.load(open(os.path.join(SRC, "results/v2_dev/best3_values.json")))
    abl = {r["program"]: r for r in json.load(open(os.path.join(SRC, "results/v2_dev/combine_ablation.json")))}
    tok = os.path.join(SRC, "results/tb_top20full/tokens")
    plain = lambda slug: json.load(open(os.path.join(tok, slug + ".json")))["bpb_all"]
    return best3, lambda slug: {
        "obsgain-nothought": best3[0]["values"].get(slug), "obsgain-easy": abl["1152f4c8_arith"]["values"].get(slug),
        "obsgain-nothought:native": best3[0]["values"].get(slug), "obsgain-easy:native": best3[1]["values"].get(slug),
        "obsgain-taskgain:native": best3[2]["values"].get(slug), "plain_ppl": plain(slug)}


def run(models, use_tokenizer):
    data = default_dir()
    docs = [d for d in trajectories(data) if d["split"] == "dev"]
    sets = pick_sets(data)
    best3, exp = expected()
    repos = panel_repos()
    allv, nfail = {}, 0
    for m in models:
        v, f = check_model(m, repos[m], docs, sets, exp, use_tokenizer)
        allv[m], nfail = v, nfail + len(f)
    return allv, nfail, best3, repos, data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(DEFAULT))
    ap.add_argument("--dumps-only", action="store_true", help="every dev model with a dump; harness renderings, no tokenizer")
    a = ap.parse_args()
    if a.dumps_only:
        tok = os.path.join(SRC, "results/tb_top20full/tokens")
        models = sorted(f[:-4] for f in os.listdir(tok) if f.endswith(".npz"))
    else:
        models = [m for m in a.models.split(",") if m]
    allv, nfail, best3, repos, data = run(models, not a.dumps_only)
    if a.dumps_only:
        sc = load_scores(os.path.join(data, "scores_dev.json"))
        print("\ndev Spearman rho (n models)\tR\tN")
        keys = ["plain_ppl"] + sorted(k for k in next(iter(allv.values())) if k != "plain_ppl")
        for k in keys:
            sign = -1.0 if k == "plain_ppl" else 1.0
            cells = []
            for t in ("R", "N"):
                ms = [m for m in models if sc[repos[m]].get(t) is not None]
                cells.append(f"{spearman([sign * allv[m][k] for m in ms], [sc[repos[m]][t] for m in ms]):+.4f} ({len(ms)})")
            print(("-" if sign < 0 else "") + k, *cells, sep="\t")
        print("harness (best3_values.json, own combine):",
              [(round(r["rho_r"], 4), round(r["rho_n"], 4)) for r in best3])
    print(f"\n{len(models)} models, {nfail} failures")
    sys.exit(1 if nfail else 0)


def test_dev_values():
    import pytest
    if not os.path.isdir(os.path.join(SRC, "results/tb_top20full/tokens")):
        pytest.skip("research repo with the dev dumps not found (AGENTPPL_SRC)")
    _, nfail, *_ = run(DEFAULT, True)
    assert nfail == 0


if __name__ == "__main__":
    main()
