# Changelog

## 0.2.0 (2026-11-27)

- AgenticPPL (the default `capped_bits` combine) is now positive capped bits: the mean over the picked ranges of
  min(2, sum of min(2, token bits)). **Lower is better.** Before 0.2.0 the same value carried a minus sign (higher was
  better); the magnitude is unchanged.
- The `prob_mean` pick sets (`obsgain-nothought`, `obsgain-taskgain`) are unchanged: probabilities in [0, 1], higher
  is better. Plain PPL is unchanged: bits per byte, lower is better.
- `agenticppl report` reads `agenticppl_version` and flips the sign of `capped_bits` values (and of a split's top-level
  `agenticppl` when its default set is `capped_bits`) in files older than 0.2.0 or without a version, so a report over
  old and new files is consistent.
- `agenticppl report --scores` prints the Spearman rho of each measure itself (plain PPL no longer negated). For bits
  (AgenticPPL, plain PPL) negative rho means the right order; for `prob_mean` sets positive rho does.

## 0.1.0 (2026-11-24)

- First release: `agenticppl score` and `agenticppl report`, pick sets `obsgain-easy` (default), `obsgain-nothought`,
  `obsgain-taskgain`.
