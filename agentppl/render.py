"""Render a trajectory with a model's own chat template, with typed byte spans.

A trajectory is a list of messages {role, content, parts}; parts = [[type, turn, char_start, char_end], ...] inside
content (task, thought, action, markup, observation). The template runs once on placeholder contents (which gives the
markup between messages) and once on the real contents. The real contents are spliced into the placeholder markup and
the result must equal the real rendering character for character, so every span holds exactly its source string:
text.encode()[start:end].decode() == content[char_start:char_end].
"""
import datetime

FIXED_DATE = datetime.datetime(2025, 8, 5)   # templates that print strftime_now("%Y-%m-%d") get a fixed date
# templates that refuse an assistant message without a thinking field (e.g. K2-Horizon) get an empty one
REASONING_EXTRA = {"assistant": {"reasoning_content": ""}}
PH = "\ue000{}\ue001"   # placeholder content: private-use characters around the message index


class _FixedDatetime(datetime.datetime):
    @classmethod
    def now(cls, tz=None):
        return FIXED_DATE


def load_tokenizer(repo):
    """A fast tokenizer (offsets are needed). Mistral repos: fix_mistral_regex=True (the shipped pre-tokenizer regex
    differs from mistral_common's)."""
    from transformers import AutoTokenizer, PreTrainedTokenizerFast
    kw = {}
    if repo.startswith("mistralai/"):
        kw["fix_mistral_regex"] = True
    try:
        tok = AutoTokenizer.from_pretrained(repo, **kw)
    except Exception:   # a config.json that this transformers cannot parse; the tokenizer files themselves load
        tok = PreTrainedTokenizerFast.from_pretrained(repo, **kw)
    if not getattr(tok, "is_fast", False):
        tok = PreTrainedTokenizerFast.from_pretrained(repo, **kw)
    if not tok.is_fast:
        raise ValueError(f"{repo}: no fast tokenizer (offsets are needed)")
    return tok


def template_kwargs(tok):
    """enable_thinking=False when the template reads it; a fixed clock when it prints the date."""
    tpl = tok.chat_template or ""
    kw = {}
    if "enable_thinking" in tpl:
        kw["enable_thinking"] = False
    if "strftime_now" in tpl:
        import transformers.utils.chat_template_utils as ctu
        ctu.datetime = _FixedDatetime
    return kw


def render(tok, kw, msgs, extra=None):
    """(text, spans [[byte_start, byte_end, type, turn], ...]); raises if the template changes any content."""
    extra = extra or {}
    ph = [{"role": m["role"], "content": PH.format(i), **extra.get(m["role"], {})} for i, m in enumerate(msgs)]
    P = tok.apply_chat_template(ph, tokenize=False, **kw)
    real = tok.apply_chat_template([{"role": m["role"], "content": m["content"], **extra.get(m["role"], {})}
                                    for m in msgs], tokenize=False, **kw)
    pieces, spans, cur, nbytes = [], [], 0, 0
    for i, m in enumerate(msgs):
        s = PH.format(i)
        if P.count(s) != 1:
            raise ValueError(f"placeholder {i} appears {P.count(s)} times")
        j = P.index(s, cur)
        pieces.append(P[cur:j])
        nbytes += len(P[cur:j].encode())
        c = m["content"]
        for typ, turn, a, b in m["parts"]:
            bs = nbytes + len(c[:a].encode())
            spans.append([bs, bs + len(c[a:b].encode()), typ, turn])
        pieces.append(c)
        nbytes += len(c.encode())
        cur = j + len(s)
    pieces.append(P[cur:])
    text = "".join(pieces)
    if text != real:
        k = next((i for i, (x, y) in enumerate(zip(text, real)) if x != y), min(len(text), len(real)))
        raise ValueError(f"template changes content near char {k}: expected {text[k-40:k+40]!r} got {real[k-40:k+40]!r}")
    return text, spans


class Renderer:
    """One model's renderer: its tokenizer, template kwargs and (if its template needs it) an empty reasoning field,
    decided once on the first trajectory."""

    def __init__(self, repo):
        self.tok = load_tokenizer(repo)
        self.kw = template_kwargs(self.tok)
        self.extra = None

    def __call__(self, msgs):
        if self.extra is None:
            try:
                out = render(self.tok, self.kw, msgs)
                self.extra = {}
                return out
            except Exception:
                out = render(self.tok, self.kw, msgs, REASONING_EXTRA)
                self.extra = REASONING_EXTRA
                return out
        return render(self.tok, self.kw, msgs, self.extra)

    def special_ids(self):
        t = self.tok
        return {i for i, x in t.added_tokens_decoder.items() if x.special} | set(t.all_special_ids)
