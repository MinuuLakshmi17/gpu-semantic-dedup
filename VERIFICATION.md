# Verification

What was actually run and checked before this repo shipped (2026-10-27).

## Verified

- **Tests: 7/7 pass** (`pytest` — top-k exact-duplicate detection with
  self-match exclusion, chunk-size agreement, single-row edge case,
  embedding determinism, normalization, pipeline I/O). The tests are
  real assertions, not placeholders.
- **End-to-end CPU smoke test, run live:** generated 100 synthetic
  listings, ran the numpy backend (top-k 5, threshold 0.88, chunk 512),
  wrote 28 directed candidate pairs to CSV. Independently checked:
  0 self-matches, 0 pairs below threshold, planted duplicates found at
  similarity 1.0.
- **Chunk-size consistency, run live:** chunk 512 vs chunk 32 produced
  byte-identical pair sets (after sorting).
- **Backend consistency, run live:** numpy backend vs PyTorch backend on
  CPU produced identical pair sets.
- **Undirected mode, run live:** 28 directed pairs canonicalized to 14
  undirected pairs with 0 reversed duplicates.
- **GPU backends fail cleanly without a GPU:** torch with device=cuda,
  cupy, and triton each raise a clear RuntimeError naming the missing
  dependency or CUDA — no confusing crash.
- **Benchmark command, run live:** 3,000 rows, numpy backend, 0.43s
  similarity time, 12,119 pairs at threshold 0.9.
- **Packaging hygiene:** numpy is the only hard dependency; PyTorch,
  CuPy, Triton, and sentence-transformers are optional extras, so a
  CPU-only install works.

## One pitch number that did not reproduce

The accompanying pitch claimed the 100-row smoke test wrote "300 directed
candidate pairs." With the documented settings (100 rows, top-k 5,
threshold 0.88) the measured count is **28**. The code is correct — the
28 pairs check out on every invariant above — but the 300 figure is not
repeated anywhere in this repo, and it should not be quoted.

## Not verified

- **All GPU execution:** CUDA paths (torch/cupy/triton), Triton kernel
  compilation, Nsight Systems/Compute profiling, and any A10G
  benchmark. The verification environment has no NVIDIA GPU
  (no `nvidia-smi`, no torch CUDA device). The GPU code was reviewed
  statically: imports are guarded, CUDA availability is checked before
  use, and chunked transfers avoid per-row host-device synchronization.
- **The illustrative 9h → 2.8h figures** from the original project brief
  are explicitly disclaimed in the README and are not claimed here.
  Run the pipeline on real GPU hardware and measure before quoting any
  speedup.
