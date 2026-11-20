# p11 — prefill/decode interleave

Local scheduling-layer change to the concurrent executor: it caps the wall clock of a
prefill execution unit and stops starving an active decode while a prefill is in flight.

Two coupled changes behind new serve options:

- **Prefill sub-step (opt-in).** When `--prefill-step-tokens` is set, the configured prefill
  chunk is advanced in smaller sub-steps of that size (128-aligned, within `[128, prefill-chunk]`)
  instead of one whole-chunk advance. Each sub-step is a bounded wall-clock unit the scheduler can
  yield a decode round after. Unset by default: the step equals the prefill chunk (no split, the
  original whole-chunk path), so C=1 pays zero prefill cost.
- **Proportional decode compensation (default).** After each prefill sub-step the executor runs a
  number of decode rounds proportional to the sub-step's measured wall clock
  `n = clamp(ceil(T / --prefill-decode-target-ms), 1, 32)` (default target 100 ms), replacing the
  previous fixed 4-round compensation. Passing `--prefill-decode-rounds N` (1..64) selects the
  fixed-count mode explicitly; leaving it unset selects proportional mode.

Why the sub-step is opt-in: at C=1 there is no in-flight decode to yield to, and splitting an
8192-token prefill into 256-token sub-steps measured **34% slower prefill** (1180.7 vs 1795.9
tok/s at a 1280 chunk, two V100s). The split only pays off under concurrent (C≥2) prefill+decode,
where it caps the per-unit GPU window; a deployment that wants that cap opts in with
`--prefill-step-tokens 256`.

At C=1 there is no decode member to compensate, so the default configuration is a numeric no-op;
the observable effect of the split is only under concurrent (C≥2) prefill+decode.

## Options

| Option | Default | Meaning |
|---|---|---|
| `--prefill-step-tokens` | (unset) | prefill sub-step, 128-aligned, within `[128, prefill-chunk]`; unset = no split (step = prefill-chunk) |
| `--prefill-decode-target-ms` | 100 | wall-clock target per prefill sub-step for proportional compensation (≥ 10) |
| `--prefill-decode-rounds` | (unset) | explicit fixed decode-compensation count (1..64); unset = proportional mode |

The `ninfer-serve` startup banner prints the effective sub-step (`prefill-step=`), and the
request log records `prefill_step_tokens` and `prefill_decode_target_ms`.

## Verification path

The dev-loop scripts that implement the tiers live outside this repository (they carry
machine-specific paths); the layers and their commands are recorded here at the tool-name level.
Measured on two V100-SXM2-16GB cards, CUDA 12.8 / SM70, `qwen3.8-27b/quasar-nvfp4`, INT8 KV, TP2.

### T0 — static wiring (no GPU)

`assert-static.sh` over the change: 16/16 assertions pass. They confirm (a) the sub-step advance
call sites are live, (b) the admission service-work quantum is projected from the same effective
sub-step the executor advances with (plan == executor), (c) the proportional compensation reads the
measured prefill wall clock and the fixed-count mode is retained, and (d) the new options thread
from `EngineOptions` through serve. Incremental build is clean.

### T2 — C=1 numeric invariance

Baseline is the closed p7 variant binary; the p11 binary defaults to no split (step = prefill
chunk), i.e. the same whole-chunk path p7 runs.

- **C=1 token identity** (`check-equivalence.sh`, 256 prompt / 16 gen, chunk 1280, int8, TP2):
  ordinary and MTP (`--spec mtp --mtp-draft-tokens 5`) both produce **17 identical token IDs**
  between p7 (whole-chunk) and p11 (sub-step 256). 0 differences.
- **Sub-step invariance** (p11 vs p11, `--prefill-step-tokens` 256 vs 1280 = whole chunk, same
  shape): **17 identical token IDs**. The bench config records the distinct step values, so this
  is a genuine split-vs-no-split comparison, not a no-op.

Cost: a handful of short bench runs, each a ~50 s weight load plus a few seconds of generation.

### T3 — concurrent discriminative A/B

C=2, one ~22.5K-token cold prefill request (distinct prompt per round, so every round is a real
cold prefill, not a prefix-cache hit) plus one 256-token long-decode request, fired concurrently.
One warmup round absorbs first-touch graph capture; three measured rounds per mode. The decode
request's aggregate output rate during the concurrent prefill is the observable
(`serve_concurrent_load.py`, streaming, decodes TTFT + per-token rate).

| Mode | Decode aggregate tok/s during concurrent prefill | % of standalone |
|---|---:|---:|
| Fixed (`--prefill-decode-rounds 4`, sub-step = chunk) | 14.08 | 23.5% |
| Standalone decode (no concurrent prefill) | 59.81 | 100% (denominator) |
| Proportional (sub-step 256, target 100 ms) | 43.14 | 72.1% |

The proportional mode sustains **43.14 tok/s** of concurrent decode versus **14.08 tok/s** for the
fixed baseline — a **3.06×** improvement, and 72% of the standalone decode rate (against 23.5% for
the fixed baseline). This clears the spec floor (concurrent decode ≥ 20% of standalone) and exceeds
the fixed baseline. The intended trade-off is visible: the proportional mode's prefill is slower
(~25 s vs ~15 s) because it yields more decode rounds to the in-flight decode.

Caveat: the decode figure is the aggregate over the 256-token window, which extends past the
prefill; the fixed baseline reads 23.5% of standalone here rather than the tighter low-single-digit
fraction seen when only the prefill-overlap window is counted. The baseline is still the starved
reference, and the binding acceptance (proportional ≥ 20% of standalone and > fixed) passes.

Cost: three server loads (~50 s each) plus twelve concurrent pairs, ≈ 6.5 min.

### T4 — interleaved end-to-end A/B

`ab-interleave.sh`-style interleaved A/B of the concurrent workload: fixed (4-round) vs
proportional, 3 blocks × 3 samples per side block-interleaved in one session (9 measured samples
per mode plus 3 warmup rounds each), declaring concurrent aggregate decode throughput + tail TTFT;
a claim is made only if the difference exceeds the repeat-measurement noise band.

| Mode | Concurrent decode aggregate tok/s (9 samples) | Per-block means |
|---|---:|---|
| Fixed (`--prefill-decode-rounds 4`) | 13.99 (σ 0.07) | 14.08 / 13.96 / 13.93 |
| Proportional (sub-step 256) | 42.84 (σ 0.17) | 42.82 / 42.88 / 42.82 |

The difference (+28.85 tok/s, +206%) exceeds the noise band (max σ 0.17 tok/s) by two orders of
magnitude, and the per-block means are stable, so the claim stands: the proportional mode delivers
**3.06× the concurrent decode throughput** of the fixed baseline (71.6% of the standalone decode
rate), at the designed cost of a slower prefill wall clock.