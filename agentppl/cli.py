"""agentppl score  --model HF_ID [--tp N] [--split dev|heldout|all] [--picks NAME] [--data DIR] [--out FILE]
agentppl report RESULT.json ... [--scores FILE]

score: renders every trajectory of the split with the model's chat template, runs one vLLM forward pass (prompt
log-probs only, no generation), and writes {model, splits: {split: {n_docs, plain_ppl, picks: {name: {agentppl,
n_ranges}}}}} as JSON. report: a table of results, and with --scores the Spearman rho of AgentPPL and of minus plain
PPL against the scores (R and N).
"""
import argparse, json, math, os, sys, time

import numpy as np

from . import __version__
from .data import SPLITS, default_dir, load_scores, pick_sets, trajectories
from .measure import agentppl_of, plain_ppl, same_spans, spearman, token_bytes

LN2 = math.log(2.0)


def log(*a):
    print(*a, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------------------------- forward pass

def prompt_bits(plp, ids):
    """bits = -log2 p(token | earlier tokens) per prompt token from vLLM prompt_logprobs (dict or flat layout);
    the first token has no context: NaN."""
    n = len(ids)
    out = np.full(n, np.nan, np.float32)
    if hasattr(plp, "start_indices") and hasattr(plp, "token_ids"):   # FlatLogprobs
        st = np.asarray(plp.start_indices, np.int64)
        tok = np.asarray(plp.token_ids, np.int64)
        lp = np.asarray(plp.logprobs, np.float64)
        for i in range(1, n):
            k = st[i] if tok[st[i]] == ids[i] else None
            out[i] = -(lp[k] if k is not None else plp[i][ids[i]].logprob) / LN2
        return out
    for i in range(1, n):
        out[i] = -plp[i][ids[i]].logprob / LN2
    return out


def forward(model, tp, prompts, dtype="bfloat16", batch=64):
    """Per-token bits of every prompt (a list of token id lists) under `model`, via vLLM."""
    import inspect
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    kw = dict(model=model, tokenizer=model, tensor_parallel_size=tp, dtype=dtype,
              max_model_len=max(len(p) for p in prompts) + 8, enable_prefix_caching=False,
              max_num_seqs=16, gpu_memory_utilization=0.88)
    try:
        from transformers import AutoConfig
        arch = (getattr(AutoConfig.from_pretrained(model), "architectures", None) or [""])[0]
    except Exception:
        arch = ""
    if "ConditionalGeneration" in arch:   # text only: no multimodal profiling
        kw["limit_mm_per_prompt"] = {"image": 0, "video": 0, "audio": 0}
    llm = LLM(**kw)
    spk = dict(max_tokens=1, prompt_logprobs=0, temperature=0.0, detokenize=False)
    if "flat_logprobs" in inspect.signature(SamplingParams).parameters:
        spk["flat_logprobs"] = True
    sp = SamplingParams(**spk)
    out, t0, ntok = [], time.time(), 0
    for k in range(0, len(prompts), batch):
        chunk = prompts[k:k + batch]
        res = llm.generate([TokensPrompt(prompt_token_ids=p) for p in chunk], sp, use_tqdm=False)
        for p, r in zip(chunk, res):
            assert list(r.prompt_token_ids) == list(p)
            out.append(prompt_bits(r.prompt_logprobs, p))
            ntok += len(p)
        log(f"{min(k + batch, len(prompts))}/{len(prompts)} docs, {ntok / (time.time() - t0):.0f} tok/s")
    return out


# ---------------------------------------------------------------------------------------------- score

def score(a):
    from .render import Renderer
    data = a.data or default_dir()
    docs = trajectories(data)
    want = list(SPLITS) if a.split == "all" else [a.split]
    docs = [d for d in docs if d["split"] in want]
    if not docs:
        raise SystemExit(f"no {a.split} trajectories in {data}" +
                         (" (held-out trajectories are not released yet)" if a.split == "heldout" else ""))
    sets = pick_sets(data, [a.picks] if a.picks else None)
    rd = Renderer(a.model)
    sid = np.array(sorted(rd.special_ids()), np.int64)
    model, prompts, special, unmatched = {}, [], [], []
    t0 = time.time()
    for d in docs:
        text, spans = rd(d["messages"])
        enc = rd.tok(text, add_special_tokens=False, return_offsets_mapping=True)
        s, e = token_bytes(text, enc["offset_mapping"])
        ok = same_spans(d["text"], d["spans"], text, spans)
        if not ok:
            unmatched.append(d["id"])
        # a content string that this tokenizer turns into a control token (e.g. "<image>" in a terminal output)
        hit = s[np.isin(np.asarray(enc["input_ids"]), sid)]
        if any(x0 <= h < x1 for h in hit for x0, x1, _, _ in spans):
            special.append(d["id"])
        model[d["id"]] = (spans if ok else None, s, e)
        prompts.append(enc["input_ids"])
    log(f"rendered and tokenized {len(docs)} docs in {time.time() - t0:.0f} s; "
        f"{sum(map(len, prompts))} tokens, longest {max(map(len, prompts))}")
    bits = dict(zip([d["id"] for d in docs], forward(a.model, a.tp, prompts, a.dtype)))
    res = {"agentppl_version": __version__, "model": a.model, "data": os.path.abspath(data),
           "splits": {}, "docs_with_special_tokens_in_content": len(special), "docs_unmatched": len(unmatched)}
    for sp in want:
        ds = [d for d in docs if d["split"] == sp]
        if not ds:
            continue
        r = {"n_docs": len(ds), "plain_ppl": plain_ppl(ds, model, bits), "picks": {}}
        for name, ps in sets.items():
            if sp in ps["splits"]:
                r["picks"][name] = agentppl_of(ds, ps["ranges"], model, bits)
        res["splits"][sp] = r
    txt = json.dumps(res, indent=1)
    if a.out:
        open(a.out, "w").write(txt + "\n")
    print(txt)


# ---------------------------------------------------------------------------------------------- report

def report(a):
    res = [json.load(open(p)) for p in a.results]
    scores = load_scores(a.scores) if a.scores else None
    for sp in SPLITS:
        rows = [(r["model"], r["splits"][sp]) for r in res if sp in r.get("splits", {})]
        if not rows:
            continue
        names = sorted({n for _, s in rows for n in s["picks"]})
        print(f"\n== {sp} ({len(rows)} models)")
        print("\t".join(["model", "plain_ppl"] + names + (["R", "N"] if scores else [])))
        for m, s in sorted(rows):
            cells = [m, f"{s['plain_ppl']:.4f}"] + [f"{s['picks'][n]['agentppl']:.4f}" if n in s["picks"] else "-" for n in names]
            if scores:
                sc = scores.get(m, {})
                cells += ["-" if sc.get(k) is None else f"{sc[k]:.3f}" for k in ("R", "N")]
            print("\t".join(cells))
        if not scores:
            continue
        cols = [("-plain_ppl", {m: -s["plain_ppl"] for m, s in rows})] + \
               [(n, {m: s["picks"][n]["agentppl"] for m, s in rows if n in s["picks"]}) for n in names]
        print("\nSpearman rho\tR (n)\tN (n)")
        for label, v in cols:
            cells = [label]
            for k in ("R", "N"):
                ms = [m for m in v if scores.get(m, {}).get(k) is not None and np.isfinite(v[m])]
                cells.append(f"{spearman([v[m] for m in ms], [scores[m][k] for m in ms]):+.3f} ({len(ms)})" if len(ms) >= 3 else f"- ({len(ms)})")
            print("\t".join(cells))
        missing = sorted(m for m, _ in rows if m not in scores)
        if missing:
            print("no score for:", ", ".join(missing))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="agentppl", description="AgentPPL: score a model on the release trajectories, or report results")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score", help="score one model (GPU, vLLM)")
    s.add_argument("--model", required=True, help="Hugging Face repo id or local path")
    s.add_argument("--tp", type=int, default=1, help="tensor parallel size")
    s.add_argument("--split", default="dev", choices=["dev", "heldout", "all"])
    s.add_argument("--picks", default=None, help="one pick set (default: every pick set in the release)")
    s.add_argument("--dtype", default="bfloat16")
    s.add_argument("--data", default=None, help="release dir (default $AGENTPPL_DATA or <repo>/release)")
    s.add_argument("--out", default=None, help="also write the JSON here")
    r = sub.add_parser("report", help="table and Spearman rho of score results")
    r.add_argument("results", nargs="+")
    r.add_argument("--scores", default=None, help="scores file, e.g. release/scores_dev.json")
    a = ap.parse_args(argv)
    score(a) if a.cmd == "score" else report(a)


if __name__ == "__main__":
    main()
