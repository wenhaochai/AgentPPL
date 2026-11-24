"""The release files (built by scripts/build_dataset.py):

  trajectories.jsonl.gz   one trajectory per line, in canonical order:
                          id (content hash), split (dev | heldout), source, source_revision, source_ref, licence,
                          task_id, teacher, n_turns, messages [{role, content, parts [[type, turn, c0, c1]]}],
                          text + spans [[b0, b1, type, turn]] (the canonical rendering, Flint's chat template)
  picks/<name>.json       a pick set: {name, splits, evaluator, program_sha256, description, ranges {doc id: [[b0, b1]]}}
                          (bytes of the doc's canonical text)
  scores_<split>.json     {"benchmark", "models": {hf repo: {"R": score, "N": score or null}}}
  MANIFEST.json           files, sizes, SHA-256, counts
"""
import glob, json, os

from .measure import read_jsonl

SPLITS = ("dev", "heldout")


def default_dir():
    return os.environ.get("AGENTPPL_DATA") or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "release")


def trajectories(data_dir):
    p = os.path.join(data_dir, "trajectories.jsonl.gz")
    if not os.path.exists(p):
        raise SystemExit(f"no {p}: build the release first (scripts/build_dataset.py) or pass --data")
    return read_jsonl(p)


def pick_sets(data_dir, names=None):
    out = {}
    for p in sorted(glob.glob(os.path.join(data_dir, "picks", "*.json"))):
        s = json.load(open(p))
        if names is None or s["name"] in names:
            out[s["name"]] = s
    if names is not None and set(names) - set(out):
        raise SystemExit(f"unknown pick set(s) {sorted(set(names) - set(out))}; have {sorted(pick_sets(data_dir))}")
    return out


def load_scores(path):
    """{model: {"R": .., "N": ..}} from a scores file (a release scores_<split>.json or a plain mapping)."""
    d = json.load(open(path))
    return d.get("models", d)
