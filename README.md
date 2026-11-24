# AgentPPL

AgentPPL predicts how an LLM ranks on Terminal-Bench 2.1 from one forward pass over fixed public agent trajectories.
It needs only the scored model: no generation, no sandbox, no judge.

**Status: private, dev results only.** The pick set to ship is not chosen yet. The held-out test (new trajectories, new
models) is pending; its numbers go in the table below when it is done.

## What it measures

1. **Trajectories.** 1,401 Terminus-2 agent trajectories on terminal tasks (Terminal-Lego-Traj-8k and terminal-wrench,
   3 teachers per task, tasks that share a 13-gram with a TB2.1 instruction removed). Every trajectory is cut to a fixed
   message prefix (at most 16k tokens under every tokenizer we checked) and typed by span: task, thought, action (the
   keystrokes the agent sends), observation (terminal output), markup.
2. **Render and score.** The scored model renders each trajectory with its own chat template, and vLLM returns the
   log-probability of every token (prompt log-probs only).
3. **Picks.** A pick set is a list of byte ranges in the trajectories. The ranges are precomputed with an evaluator
   model, Flint (`RecursiveSI/flint`), and shipped as data. All three current sets pick command tokens that Flint
   predicts much better when it can see the earlier terminal output (the agent reads them off the screen). The ranges are
   mapped to any model's rendering by (span, offset in span), since every rendering holds the same span texts.
4. **AgentPPL** of a model on a pick set is the arithmetic mean, over the ranges, of the probability the model gives
   each range's text:

   p(range) = 2^-bits, with bits = (sum of the bits of the model's tokens covering the range) × (range bytes / bytes of
   those tokens).

   A token that sticks out of a range counts only for its share, so the value does not depend on the tokenizer.
   Higher AgentPPL = higher predicted TB2.1 rank.
5. **Plain PPL**, for comparison: bits per byte over all tokens of the model's rendering.

The arithmetic mean of probabilities matters: the mean of bits (the geometric mean of probabilities) on the same picks
gives dev rho_R +0.64 instead of +0.90, because one confidently wrong token can add 10-40 bits to a mean of bits but
at most one miss to a mean of probabilities.

## Results

Spearman rho with Terminal-Bench 2.1 (Artificial Analysis, Terminus 2). R = reasoning-mode score, N = non-reasoning
score (a model with one AA score has it as both).

| measure | bytes used | dev rho_R (25 models) | dev rho_N (20 models) | held-out rho_R | held-out rho_N |
|---|---|---|---|---|---|
| AgentPPL, `obsgain-nothought` | 0.73% | +0.903 | +0.938 | _pending_ | _pending_ |
| AgentPPL, `obsgain-easy` | 0.30% | +0.894 | +0.937 | _pending_ | _pending_ |
| AgentPPL, `obsgain-taskgain` | 0.51% | +0.897 | +0.947 | _pending_ | _pending_ |
| plain PPL (minus bits per byte) | 100% | −0.292 | −0.289 | _pending_ | _pending_ |

- "bytes used" is the share of all bytes of the canonical rendering (31.65 M bytes); every picked byte is in a command.
- **The dev numbers are optimistic.** The pick rules were found by agents that saw the dev rho of these 25 models on
  these 1,401 trajectories. Only the held-out columns test the method.
- Dev models: 2-14B and 59-128B open instruct models (`release/scores_dev.json`). Numbers from `tests/test_dev_values.py
  --dumps-only`.

## Install and run

```bash
pip install -e .[gpu]                 # numpy, transformers, jinja2; vllm for scoring
python scripts/build_dataset.py --src /path/to/token-ppl --out release   # or point AGENTPPL_DATA at a release dir

agentppl score --model Qwen/Qwen3.5-4B --tp 1 --split dev --out qwen3.5-4b.json
agentppl report *.json --scores release/scores_dev.json
```

`score` writes, per split, `plain_ppl`, and per pick set `agentppl` and `n_ranges` (ranges with a scored token):

```json
{"model": "Qwen/Qwen3.5-4B",
 "splits": {"dev": {"n_docs": 1401, "plain_ppl": 1.6362,
                    "picks": {"obsgain-nothought": {"agentppl": 0.5990, "n_ranges": 55995, "n_ranges_in_set": 55995}}}},
 "docs_with_special_tokens_in_content": 0, "docs_unmatched": 0}
```

- `--picks NAME` scores one pick set (default: all). `--split heldout|all` works once held-out trajectories are in the
  release.
- `docs_with_special_tokens_in_content`: trajectories where the model's tokenizer turns a content string (for example
  `<image>` in a terminal output) into a control token. `docs_unmatched`: renderings whose spans differ from the
  canonical ones (their ranges get no tokens). Both should be 0.
- One pass over the dev trajectories is about 9-10 M tokens.
- Chat templates: thinking is turned off (`enable_thinking=False`) when the template reads it; a template that needs a
  reasoning field gets an empty one; a template that prints the date gets 2025-08-05.

`report` prints a table of results and, with `--scores`, the Spearman rho of AgentPPL and of minus plain PPL against
R and N. A scores file is `{"models": {model: {"R": x, "N": y}}}` (or the inner mapping alone), keyed by the `--model`
string used in `score`.

## Release files

Built by `scripts/build_dataset.py` (writes files only; nothing is uploaded). Formats are in `agentppl/data.py`.

| file | size | content |
|---|---|---|
| `trajectories.jsonl.gz` | 9.3 MB | 1,401 dev trajectories: id, split, source dataset, revision, source reference, licence, task, teacher, messages with typed parts, canonical rendering (Flint's template) with typed byte spans and turns |
| `picks/obsgain-nothought.json` | 0.80 MB | 55,995 ranges, 232,194 bytes |
| `picks/obsgain-easy.json` | 0.34 MB | 23,175 ranges, 94,720 bytes |
| `picks/obsgain-taskgain.json` | 0.53 MB | 36,813 ranges, 160,469 bytes |
| `scores_dev.json` | 2 KB | TB2.1 R and N of the 25 dev models |
| `MANIFEST.json` | 4 KB | sizes, SHA-256, counts |

- Trajectory ids are content hashes: SHA-256 of (source dataset, teacher, messages), first 16 hex digits. The teacher
  is in the hash because the Terminal-Lego GLM-5 and DeepSeek-V3.2 files hold 42 identical trajectories (same task,
  same messages); the dev set keeps both copies, as the dev measurement did (`MANIFEST.json` lists them).
- The pick sets, what each picks, and the search run and program each comes from:

| set | program (SHA-256 prefix) | picks |
|---|---|---|
| `obsgain-nothought` | 466edc35 | command tokens whose Flint loss drops > 0.5 bits when the earlier terminal output is visible, measured with the agent's thoughts removed on both sides |
| `obsgain-easy` | 1152f4c8 | command tokens of turns ≥ 1 whose Flint loss drops > 0.5 bits with the output visible, and that Flint predicts in < 4 bits |
| `obsgain-taskgain` | b7f43f6c | command tokens of turns ≥ 1 whose Flint loss drops > 0.5 bits with the output visible; in a trajectory with none, turn-0 command tokens where the task description helps Flint by > 1 bit |

- **Held-out split: not in this release.** The held-out trajectories and their picks stay sealed until the held-out
  judgement. The hook is `heldout()` in `scripts/build_dataset.py` (`--heldout` refuses for now); it will add records
  with `"split": "heldout"` and each pick set's held-out ranges to the same files.

## Test

`tests/test_dev_values.py` checks the toolkit against the research harness on dev models, from their stored per-token
outputs (CPU only). For each model, the toolkit renders every trajectory with the model's template and tokenizes it;
the text, spans and token byte offsets must equal the harness's. Then it maps each pick set onto the model's tokens and
combines the stored bits; AgentPPL must equal the harness's values to 1e-9, each source program's own combine run on the
toolkit's rows must equal the harness's value of that program to 1e-9, and plain PPL must equal the stored summary to
1e-6.

```bash
AGENTPPL_SRC=/path/to/token-ppl python tests/test_dev_values.py              # 5 models, full path
AGENTPPL_SRC=/path/to/token-ppl python tests/test_dev_values.py --dumps-only # all 25, the harness's renderings
```

Last run (2026-11-24): 5 of 5 models on the full path and 25 of 25 from the stored outputs, 0 failures.

## Licences and attribution

- **Code:** Apache-2.0 (`LICENSE`).
- **Trajectories:** derived from two Apache-2.0 datasets (licences read on their Hugging Face cards, 2026-11-24).
  Each record names its source, revision and source reference. Redistribution must keep the attribution and licence.
  - Terminal-Lego-Traj-8k, LegoX Team, <https://huggingface.co/datasets/Lego-X/Terminal-Lego-Traj-8k> (revision
    02279e0). Its card notes that third-party code and task materials inside it stay under their own licences.
  - Terminal Wrench, <https://huggingface.co/datasets/few-sh/terminal-wrench> (trajectories, revision 6ab64c4; index
    `few-sh/terminal-wrench-trajectories`, revision 6224497). Baseline (non-hack) trajectories of SETA tasks
    (`camel-ai/seta-env`) only.

    ```bibtex
    @misc{terminalwrench2026,
      title  = {Terminal Wrench: A Dataset of 331 Reward-Hackable Environments and 3,632 Exploit Trajectories},
      author = {Ivan Bercovich and Ivgeni Segal and Kexun Zhang and Shashwat Saxena and Aditi Raghunathan and Ziqian Zhong},
      year   = {2026},
      url    = {https://github.com/few-sh/terminal-wrench},
    }
    ```
- **Picks (derived from Flint):** `RecursiveSI/flint` is released under Apache-2.0 (its Hugging Face card, checked
  2026-11-24: "Flint is released under the Apache 2.0 license. Hosted API terms and data practices are separate from the
  weight license."). The picks are byte offsets into the trajectories, computed from Flint's per-token losses on local
  BF16 weights (no hosted API). Apache-2.0 sets no limit on the use of a model's outputs, and the picks hold no Flint
  weights or code, so they may be shipped; we credit Flint as the evaluator.
- **Scores:** Terminal-Bench 2.1 results from Artificial Analysis (Terminus 2, 89 tasks, 3 runs each). Cite them as
  Artificial Analysis when you reuse `scores_dev.json`.

## Known limits

- The vLLM path (`agentppl score`) has not run on a GPU in this repository yet; its CPU parts (rendering, tokenizing,
  mapping, combining) are what the test checks, and `score` itself was checked end to end with stored bits in place of
  vLLM.
- Models whose tokenizer or weights need `trust_remote_code`, extra vLLM arguments, or a chat template from another
  repository (for example Magistral, whose official repository has no Hugging Face template) are not handled.
