"""From one model's per-token bits on its own rendering of the trajectories to AgentPPL and plain PPL (numpy only).

Coordinates. A pick is a byte range [start, end) of a trajectory's canonical rendering (`text` in trajectories.jsonl,
the rendering with the evaluator Flint's template). Every rendering holds the same non-markup spans (task, thought,
action, observation) with the same text in the same order; only the template markup between them differs. So a range
is mapped to a model's rendering by (non-markup span, offset in it). A range outside a non-markup span is dropped.

Rows of a range: the model's tokens that start inside the span holding the range and overlap the range.

Combines (each pick set names its own; higher = predicted higher TB2.1; ranges with no scored row are left out, as a
doc's first token has no loss):
  capped_bits  (the default set's) minus the mean over ranges of min(2, sum over the range's rows of min(2, bits)):
               a confidently wrong token costs at most 2 bits.
  prob_mean    the arithmetic mean over ranges of the probability 2^-bits, bits = (sum of the rows' bits) * (range
               bytes / the rows' bytes), so a token that sticks out of the range counts only for its share.

Plain PPL: bits per byte over all tokens of the model's rendering (template markup included), tokens with a loss only.
Lower = lower loss.
"""
import gzip, json, os

import numpy as np


def open_text(path):
    return gzip.open(path, "rt", encoding="utf-8") if path.endswith(".gz") else open(path, encoding="utf-8")


def read_jsonl(path):
    with open_text(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def nonmarkup(spans):
    """The non-markup spans in order, as (start, end, type, turn)."""
    return [(int(s[0]), int(s[1]), s[2], s[3]) for s in sorted(spans, key=lambda x: x[0]) if s[2] != "markup"]


def char_to_byte(text):
    """Byte offset of every character offset (len(text) + 1 entries)."""
    cb = np.zeros(len(text) + 1, dtype=np.int64)
    np.cumsum([len(ch.encode("utf-8")) for ch in text], out=cb[1:])
    return cb


def token_bytes(text, offsets):
    """(start, end) byte offsets of tokens from a tokenizer's character offset mapping."""
    cb = char_to_byte(text)
    off = np.asarray(offsets, np.int64).reshape(-1, 2)
    return cb[off[:, 0]], cb[off[:, 1]]


def same_spans(canon_text, canon_spans, text, spans):
    """True if the two renderings hold the same non-markup spans (type, turn, text) in the same order."""
    a, b = nonmarkup(canon_spans), nonmarkup(spans)
    ca, cb = canon_text.encode(), text.encode()
    return len(a) == len(b) and all(x[2] == y[2] and x[3] == y[3] and ca[x[0]:x[1]] == cb[y[0]:y[1]] for x, y in zip(a, b))


def range_rows(canon_spans, model_spans, ranges, start, end):
    """For ranges [[a, b), ...] (canonical bytes) of one doc: (kept, rows, rid).
    kept: bool per range (inside one non-markup span); rows: token indices; rid: the kept range of each row (index
    among the kept ranges of this doc). model_spans=None: the doc's rendering did not match; kept ranges get no rows."""
    r = np.asarray(ranges, np.int64).reshape(-1, 2)
    a, b = r[:, 0], r[:, 1]
    cn = nonmarkup(canon_spans)
    cs = np.array([x[0] for x in cn], np.int64)
    ce = np.array([x[1] for x in cn], np.int64)
    j = np.searchsorted(cs, a, "right") - 1
    jj = np.maximum(j, 0)
    kept = (j >= 0) & (a >= cs[jj]) & (b <= ce[jj]) if len(cs) else np.zeros(len(a), bool)
    if model_spans is None or not kept.any():
        return kept, np.zeros(0, np.int64), np.zeros(0, np.int64)
    ms = np.array([x[0] for x in nonmarkup(model_spans)], np.int64)[j[kept]]
    A = ms + (a[kept] - cs[j[kept]])
    B = A + (b[kept] - a[kept])
    start, end = np.asarray(start, np.int64), np.asarray(end, np.int64)
    lo = np.maximum(np.searchsorted(end, A, "right"), np.searchsorted(start, ms, "left"))
    hi = np.searchsorted(start, B, "left")
    n = np.maximum(hi - lo, 0)
    rid = np.repeat(np.arange(len(A)), n)
    first = np.repeat(lo - np.concatenate([[0], np.cumsum(n)[:-1]]), n)
    return kept, first + np.arange(int(n.sum())), rid


CAP = 2.0


def capped_bits(bits, rid, n):
    """Minus the mean over ranges of min(CAP, sum of min(CAP, row bits)). Returns (value, ranges with a scored row)."""
    ok = ~np.isnan(bits)
    has = np.bincount(rid[ok], minlength=n) > 0
    if not has.any():
        return float("nan"), 0
    tot = np.bincount(rid[ok], weights=np.minimum(np.asarray(bits, np.float64)[ok], CAP), minlength=n)
    return -float(np.minimum(tot[has], CAP).mean()), int(has.sum())


def agentppl(bits, nbytes, rid, range_bytes):
    """Arithmetic mean over ranges of 2^-(range bits), range bits = sum of row bits * range bytes / row bytes.
    Returns (value, number of ranges with a scored row)."""
    n = len(range_bytes)
    ok = ~np.isnan(bits)
    raw_bits = np.bincount(rid[ok], weights=np.asarray(bits, np.float64)[ok], minlength=n)
    raw_bytes = np.bincount(rid[ok], weights=np.asarray(nbytes, np.float64)[ok], minlength=n)
    has = raw_bytes > 0
    est = raw_bits[has] * (np.asarray(range_bytes, np.float64)[has] / raw_bytes[has])
    if not has.any():
        return float("nan"), 0
    return float((2.0 ** (-np.clip(est, 0.0, 60.0))).mean()), int(has.sum())


def gather(docs, picks, model):
    """One model's rows of one pick set over docs (in doc order). docs: canonical records (id, text, spans); picks:
    {doc id: [[a, b], ...]}; model: {doc id: (spans or None, token start, token end)}.
    Returns ([(doc id, rows, rid as a range index over all docs)], range_bytes of the kept ranges, dropped ranges)."""
    out, rbytes, dropped, base = [], [], 0, 0
    for d in docs:
        rs = picks.get(d["id"])
        if not rs:
            continue
        spans, start, end = model[d["id"]]
        kept, rows, rid = range_rows(d["spans"], spans, rs, start, end)
        r = np.asarray(rs, np.int64).reshape(-1, 2)[kept]
        dropped += int((~kept).sum())
        out.append((d["id"], rows, rid + base))
        rbytes.append(r[:, 1] - r[:, 0])
        base += len(r)
    return out, (np.concatenate(rbytes) if rbytes else np.zeros(0, np.int64)), dropped


def agentppl_of(docs, picks, model, bits, combine="prob_mean"):
    """AgentPPL of one model and pick set. bits: {doc id: float array per token}. Returns a dict."""
    doc_rows, rbytes, dropped = gather(docs, picks, model)
    b = [np.asarray(bits[i], np.float64)[rows] for i, rows, _ in doc_rows]
    nb = [(np.asarray(model[i][2])[rows] - np.asarray(model[i][1])[rows]) for i, rows, _ in doc_rows]
    rid = [r for _, _, r in doc_rows]
    cat = lambda xs, dt: np.concatenate(xs).astype(dt) if xs else np.zeros(0, dt)
    if combine == "capped_bits":
        v, n = capped_bits(cat(b, np.float64), cat(rid, np.int64), len(rbytes))
    elif combine == "prob_mean":
        v, n = agentppl(cat(b, np.float64), cat(nb, np.float64), cat(rid, np.int64), rbytes)
    else:
        raise ValueError(f"unknown combine {combine!r}")
    return {"agentppl": v, "combine": combine, "n_ranges": n, "n_ranges_in_set": int(len(rbytes) + dropped)}


def plain_ppl(docs, model, bits):
    """Bits per byte over every token with a loss in the model's renderings of docs."""
    tot_bits = tot_bytes = 0.0
    for d in docs:
        _, s, e = model[d["id"]]
        b = np.asarray(bits[d["id"]], np.float64)
        ok = ~np.isnan(b)
        tot_bits += float(b[ok].sum())
        tot_bytes += float((np.asarray(e) - np.asarray(s))[ok].sum())
    return tot_bits / tot_bytes if tot_bytes else float("nan")


def ranks(v):
    """Average ranks (ties share their mean rank), 1-based."""
    v = np.asarray(v, float)
    o = np.argsort(v, kind="mergesort")
    r = np.empty(len(v))
    sv = v[o]
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and sv[j + 1] == sv[i]:
            j += 1
        r[o[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(a, b):
    a, b = ranks(a), ranks(b)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])
