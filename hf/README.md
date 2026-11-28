---
license: apache-2.0
pretty_name: AgenticPPL
size_categories:
- 1K<n<10K
tags:
- agents
- terminal-bench
- evaluation
- perplexity
configs:
- config_name: trajectories
  data_files:
  - split: dev
    path: trajectories.jsonl.gz
---

# AgenticPPL

Fixed public agent trajectories and precomputed byte ranges for **AgenticPPL**, a cheap predictor of an LLM's
Terminal-Bench 2.1 rank from one forward pass. Code: <https://github.com/wenhaochai/AgenticPPL>.

On 960 trajectories of new tasks and 15 open instruct models that the search never saw, AgenticPPL ranks the models with
Spearman rho −0.79 against their TB2.1 scores, where plain perplexity on the same text gets +0.50. Both are bits (lower
is better), so negative rho means the right order. A model's score is
its Artificial Analysis TB2.1 score in reasoning mode at the highest effort, or AA's single score when the model has one
mode (`R` in `scores_dev.json`).

## Files

| file | content |
|---|---|
| `trajectories.jsonl.gz` | 2,361 Terminus-2 trajectories, one JSON object per line. The `split` field is `dev` (1,401) or `heldout` (960). Fields: `id` (content hash), `split`, `source`, `source_revision`, `source_ref`, `licence`, `task_id`, `teacher`, `n_turns`, `messages` (role, content, typed parts), `text` + `spans` (the canonical rendering with Flint's chat template; spans are `[byte_start, byte_end, type, turn]`) |
| `picks/obsgain-easy.json` | the default pick set: byte ranges `[start, end)` of each trajectory's canonical `text`, keyed by trajectory id; combine `capped_bits` |
| `picks/obsgain-nothought.json`, `picks/obsgain-taskgain.json` | two extra pick sets; combine `prob_mean` |
| `scores_dev.json` | Terminal-Bench 2.1 scores (Artificial Analysis) of the 25 dev models; use `R` |
| `MANIFEST.json` | sizes, SHA-256, counts |

Note: in the viewer, every row shows under the `dev` split. Filter on the `split` field to get the held-out trajectories.

## Use

```bash
pip install git+https://github.com/wenhaochai/AgenticPPL
huggingface-cli download wchai/AgenticPPL --repo-type dataset --local-dir release
AGENTICPPL_DATA=release agenticppl score --model Qwen/Qwen3.5-4B --split heldout
```

AgenticPPL (default) is the mean over the picked ranges of the model's capped bits on each range: each token's bits
capped at 2 and each range's sum capped at 2. Lower is better. The ranges were picked with Flint's per-token losses:
they are command tokens the agent reads off the earlier terminal output.

## Sources and licences

Composition (all Terminus-2 runs; no teacher is among the scored models):

| source | tasks (dev / held-out) | teachers | trajectories (dev / held-out) |
|---|---|---|---|
| Terminal-Lego-Traj-8k: StackOverflow questions turned into Docker tasks | 292 / 288 | Claude Opus 4.6, GLM-5, DeepSeek-V3.2 | 876 / 864 |
| terminal-wrench: successful baseline runs on SETA tasks | 175 / 54 | Claude Opus 4.6, Gemini 3.1 Pro, GPT-5.4 | 525 / 96 |

- Terminal-Lego-Traj-8k (LegoX Team, Apache-2.0), <https://huggingface.co/datasets/Lego-X/Terminal-Lego-Traj-8k>,
  revision 02279e0. Third-party code and task materials inside it stay under their own licences.
- Terminal Wrench (Apache-2.0), <https://huggingface.co/datasets/few-sh/terminal-wrench>, revision 6ab64c4; baseline
  (non-hack) trajectories of SETA tasks only.

  ```bibtex
  @misc{terminalwrench2026,
    title  = {Terminal Wrench: A Dataset of 331 Reward-Hackable Environments and 3,632 Exploit Trajectories},
    author = {Ivan Bercovich and Ivgeni Segal and Kexun Zhang and Shashwat Saxena and Aditi Raghunathan and Ziqian Zhong},
    year   = {2026},
    url    = {https://github.com/few-sh/terminal-wrench},
  }
  ```
- Picks: computed with Recursive Flint (<https://huggingface.co/RecursiveSI/flint>, Apache-2.0) on local weights.
- Scores: Terminal-Bench 2.1 results from Artificial Analysis (Terminus 2). Only the 25 dev models are included.
- This dataset: Apache-2.0. Each trajectory record names its source, revision and source reference.
