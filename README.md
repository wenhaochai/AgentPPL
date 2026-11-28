# AgenticPPL

AgenticPPL predicts how an LLM ranks on Terminal-Bench 2.1 from one forward pass over fixed public agent trajectories.
It needs only the scored model: no generation, no sandbox, no judge.

On 960 trajectories of new tasks and 15 open instruct models that the search never saw, AgenticPPL ranks the models with
Spearman rho −0.79 against their TB2.1 scores, where plain perplexity on the same text gets +0.50. Both are bits (lower
is better), so negative rho means the right order.

Dataset: <https://huggingface.co/datasets/wchai/AgenticPPL>

## What it measures

1. **Trajectories.** Terminus-2 agent trajectories on terminal tasks from Terminal-Lego-Traj-8k and terminal-wrench,
   3 teachers per task, with every task that shares a 13-gram with a TB2.1 instruction removed. Two splits:
   `dev` (1,401 trajectories, the tasks the picks were searched on) and `heldout` (960 trajectories of other tasks).
   Each trajectory is cut to a fixed message prefix (at most 16k tokens under every tokenizer we checked) and typed by
   span: task, thought, action (the keystrokes the agent sends), observation (terminal output), markup.
2. **Render and score.** The scored model renders each trajectory with its own chat template. vLLM returns the
   log-probability of every token (prompt log-probs only).
3. **Picks.** A pick set is a list of byte ranges in the trajectories. The ranges were computed once with an evaluator
   model, Flint (`RecursiveSI/flint`), and are shipped as data. The default set, `obsgain-easy`, picks command tokens
   of turn 1 or later that Flint predicts much better when it can see the earlier terminal output (loss drop > 0.5
   bits) and that Flint finds easy in full context (< 4 bits): text the agent reads off the screen. The ranges are
   mapped to any model's rendering by (span, offset in span), because every rendering holds the same span texts.
4. **AgenticPPL** of a model is the mean over the picked ranges of the model's capped bits on each range: each
   token's bits capped at 2 and each range's sum capped at 2. With the cap, one confidently wrong token costs at most
   2 bits. Lower is better: lower AgenticPPL = higher predicted TB2.1 rank.
5. **Plain PPL**, for comparison: bits per byte over all tokens of the model's rendering.

## Results

rho of each measure with TB2.1; both are bits (lower is better), so negative rho means the measure orders models the
right way. A model's score is its Artificial Analysis TB2.1 score in reasoning mode at the highest effort,
or AA's single score when the model has one mode.

| measure | dev trajectories × 25 dev models | held-out trajectories × 25 dev models | held-out trajectories × 15 held-out models |
|---|---|---|---|
| **AgenticPPL** (`obsgain-easy`, default) | −0.890 | −0.854 | **−0.788** |
| plain PPL, bits per byte | +0.29 | +0.288 | +0.495 |

- The held-out models are 15 open instruct models of 21-37B total parameters. The 25 dev models are 2-14B and 59-128B,
  so the held-out band is new to the search, and its narrow size range leaves model size little of the ranking to
  explain. Held-out model names, scores and per-model values are not released.
- On held-out trajectories × held-out models, the AgenticPPL rho is 1.28 further toward the right order than the plain
  PPL rho (paired bootstrap over the models, 10,000 draws, 95% interval [0.57, 1.79]).
- **How the default was chosen.** Three agent search runs each froze their best program on dev. All three were
  measured once, pre-registered, on both held-out cells. `obsgain-easy` (run 05dc517c, program 1152f4c8) was chosen
  as the default after that test. The other two (higher is better) got rho +0.695 (`obsgain-nothought`) and +0.683
  (`obsgain-taskgain`, with its own top-20 rank combine) on held-out trajectories × held-out models. So the −0.788 is
  the best of three on that cell, not a fresh draw.
- The dev numbers are optimistic: the agents that wrote the pick rules saw the dev rho of the 25 dev models on the dev
  trajectories.

Extra pick sets (in the release, scored with the probability mean: the arithmetic mean over ranges of 2^-bits, bits
scaled by range bytes / covering-token bytes; a probability, so higher is better and positive rho is the right
order):

| set | dev × dev | held-out traj. × dev models |
|---|---|---|
| `obsgain-nothought` (466edc35) | +0.903 | +0.869 |
| `obsgain-taskgain` (b7f43f6c) | +0.897 | +0.893 |

Their held-out-model numbers with this combine were not measured. `tests/test_dev_values.py --dumps-only` prints all
dev-model columns.

## Install and run

```bash
pip install -e .[gpu]                 # numpy, transformers, jinja2; vllm for scoring
huggingface-cli download wchai/AgenticPPL --repo-type dataset --local-dir release   # or AGENTICPPL_DATA=DIR

agenticppl score --model Qwen/Qwen3.5-4B --tp 1 --split heldout --out qwen3.5-4b.json
agenticppl report *.json --scores release/scores_dev.json
```

`score` writes, per split, the default AgenticPPL first, then plain PPL and every pick set:

```json
{"model": "Qwen/Qwen3.5-4B",
 "splits": {"heldout": {"n_docs": 960, "picks_name": "obsgain-easy", "agenticppl": 0.7755, "n_ranges": 10568,
                        "plain_ppl": 1.5233,
                        "picks": {"obsgain-easy": {"agenticppl": 0.7755, "combine": "capped_bits", "n_ranges": 10568},
                                  "obsgain-nothought": {"agenticppl": 0.5850, "combine": "prob_mean", "n_ranges": 28123}}}},
 "docs_with_special_tokens_in_content": 0, "docs_unmatched": 0}
```

- `--split dev|heldout|all`. `--picks NAME` scores one pick set.
- `docs_with_special_tokens_in_content`: trajectories where the model's tokenizer turns a content string (for example
  `<image>` in a terminal output) into a control token. `docs_unmatched`: renderings whose spans differ from the
  canonical ones (their ranges get no tokens). Both should be 0.
- One pass is about 9.3 M tokens (dev) or 6.2 M tokens (held-out) for a typical tokenizer.
- Chat templates: thinking is turned off (`enable_thinking=False`) when the template reads it. A template that needs a
  reasoning field gets an empty one. A template that prints the date gets 2025-08-05.
- `report` prints a table of results and, with `--scores`, the Spearman rho of each measure itself against the score.
  Lower bits should rank higher, so for AgenticPPL and plain PPL negative rho means the right order; for the
  probability-mean sets positive rho does. Result files written before 0.2.0 carry AgenticPPL with a minus sign;
  `report` reads `agenticppl_version` and flips them, so old and new files mix. A scores file is
  `{"models": {model: {"R": x}}}`, keyed by the `--model` string used in `score`; `R` is the score described under
  Results.

## Trajectories

Composition (all Terminus-2 runs; no teacher is among the scored models):

| source | tasks (dev / held-out) | teachers | trajectories (dev / held-out) |
|---|---|---|---|
| Terminal-Lego-Traj-8k: StackOverflow questions turned into Docker tasks | 292 / 288 | Claude Opus 4.6, GLM-5, DeepSeek-V3.2 | 876 / 864 |
| terminal-wrench: successful baseline runs on SETA tasks | 175 / 54 | Claude Opus 4.6, Gemini 3.1 Pro, GPT-5.4 | 525 / 96 |

## Release files

Built by `scripts/build_dataset.py` from the research repository. Formats are in `agenticppl/data.py`.

| file | size | content |
|---|---|---|
| `trajectories.jsonl.gz` | 15.4 MB | 2,361 trajectories (1,401 dev, 960 held-out): id, split, source dataset, revision, source reference, licence, task, teacher, messages with typed parts, canonical rendering (Flint's template) with typed byte spans and turns |
| `picks/obsgain-easy.json` | 0.50 MB | default: 23,175 dev + 10,568 held-out ranges; combine `capped_bits` |
| `picks/obsgain-nothought.json` | 1.20 MB | 55,995 + 28,123 ranges; combine `prob_mean` |
| `picks/obsgain-taskgain.json` | 0.84 MB | 36,813 + 21,492 ranges; combine `prob_mean` |
| `scores_dev.json` | 2 KB | TB2.1 scores of the 25 dev models (Artificial Analysis; use `R`, the score described under Results) |
| `MANIFEST.json` | 10 KB | sizes, SHA-256, counts |

- Trajectory ids are content hashes: SHA-256 of (source dataset, teacher, messages), first 16 hex digits. The teacher
  is in the hash because the Terminal-Lego GLM-5 and DeepSeek-V3.2 files hold identical trajectories for some tasks
  (42 pairs in each split). Both copies stay in the data, as in the measurements above. `MANIFEST.json` lists them.
- Each pick set is the output of a frozen program over Flint's per-token losses on variants of each trajectory. The
  held-out ranges come from the same programs run on Flint's passes over the held-out trajectories. Their counts equal
  the counts of the pre-registered held-out run.

| set | program (SHA-256 prefix) | picks |
|---|---|---|
| `obsgain-easy` | 1152f4c8 | command tokens of turns ≥ 1 whose Flint loss drops > 0.5 bits with the terminal output visible, and that Flint predicts in < 4 bits |
| `obsgain-nothought` | 466edc35 | command tokens whose Flint loss drops > 0.5 bits with the output visible, with the agent's thoughts removed on both sides |
| `obsgain-taskgain` | b7f43f6c | command tokens of turns ≥ 1 whose Flint loss drops > 0.5 bits with the output visible; in a trajectory with none, turn-0 command tokens where the task description helps Flint by > 1 bit |

## Test

`tests/test_dev_values.py` checks the toolkit against the research harness from the dev models' stored per-token
outputs (CPU only), on both splits. For each model, the toolkit renders every trajectory with the model's template and
tokenizes it; the text, spans and token byte offsets must equal the harness's. Then it maps each pick set onto the
model's tokens and combines the stored bits. AgenticPPL and each source program's own combine must equal the harness's
values to 1e-9, and plain PPL must equal the stored summary to 1e-6. On the held-out split the reference is the
pre-registered held-out run on the dev models.

```bash
AGENTICPPL_SRC=/path/to/token-ppl python tests/test_dev_values.py --split dev               # 5 models, full path
AGENTICPPL_SRC=/path/to/token-ppl python tests/test_dev_values.py --split heldout --dumps-only  # all 25 dev models
```

Last run (2026-11-24): 5 of 5 models on the full path; 25 of 25 dev models on each split from the stored outputs;
0 failures. The GPU smoke test is at the end of this file.

## Licences and attribution

- **Code:** Apache-2.0 (`LICENSE`).
- **Trajectories:** derived from two Apache-2.0 datasets (licences read on their Hugging Face cards, 2026-11-24).
  Each record names its source, revision and source reference. Redistribution must keep the attribution and licence.
  - Terminal-Lego-Traj-8k, LegoX Team, <https://huggingface.co/datasets/Lego-X/Terminal-Lego-Traj-8k> (revision
    02279e0). Its card notes that third-party code and task materials inside it stay under their own licences.
  - Terminal Wrench, <https://huggingface.co/datasets/few-sh/terminal-wrench> (trajectories, revision 6ab64c4; index
    `few-sh/terminal-wrench-trajectories`, revision 6224497). Only baseline (non-hack) trajectories of SETA tasks
    (`camel-ai/seta-env`).

    ```bibtex
    @misc{terminalwrench2026,
      title  = {Terminal Wrench: A Dataset of 331 Reward-Hackable Environments and 3,632 Exploit Trajectories},
      author = {Ivan Bercovich and Ivgeni Segal and Kexun Zhang and Shashwat Saxena and Aditi Raghunathan and Ziqian Zhong},
      year   = {2026},
      url    = {https://github.com/few-sh/terminal-wrench},
    }
    ```
- **Evaluator and picks:** Recursive Flint, <https://huggingface.co/RecursiveSI/flint>, released under Apache-2.0 (its
  card, checked 2026-11-24: "Flint is released under the Apache 2.0 license. Hosted API terms and data practices are
  separate from the weight license."). The picks are byte offsets into the trajectories, computed from Flint's
  per-token losses on local BF16 weights (no hosted API). They hold no Flint weights or code, and Apache-2.0 sets no
  limit on the use of a model's outputs.
- **Scores:** the Terminal-Bench 2.1 scores in `scores_dev.json` come from Artificial Analysis (Terminus 2, 89 tasks,
  3 runs each). Only the 25 dev models' scores are included. Cite Artificial Analysis when you reuse them.

## Known limits

- Models whose tokenizer or weights need `trust_remote_code`, extra vLLM arguments, or a chat template from another
  repository (for example Magistral, whose official repository has no Hugging Face template) are not handled.
- The signal may depend on the Terminus-2 format: all trajectories use it.

## GPU smoke test

`scripts/smoke_gpu.sh` (wq w4532, 2026-11-24): `agenticppl score --split dev` on 8 dev models (Qwen3.5-2B/4B/9B,
Ministral-3-8B/14B, gemma-4-12B, granite-4.2-3b/8b), one H100 each, one vLLM pass per model (2-9 min at
22-110k tok/s). `scripts/smoke_compare.py` compared plain PPL and the three pick sets with the same measures on the
stored harness passes: all 32 values within 1e-3, max |diff| 1.4e-4 (Qwen3.5-9B plain PPL). That is bf16 run-to-run
noise; 4 of the 8 models matched to 1e-7 or better. The run used the probability-mean combine for every set, from
before the default changed. The default's capped-bits combine works on the same per-token bits and is checked against
the harness by the CPU test.
