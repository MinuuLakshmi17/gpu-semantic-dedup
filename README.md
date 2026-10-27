# GPU Semantic Deduplication Lab

A reproducible, hardware-aware reference pipeline for finding near-duplicate product listings using embeddings and chunked cosine top-k retrieval. Includes CPU baseline, optional PyTorch/CuPy/Triton paths, synthetic data generation, Parquet output, profiling recipes, and correctness tests.

**Important honesty note:** This repository is a runnable engineering lab, not evidence that a 12-million-row workload was run here. Performance figures such as 9h → 2.8h are illustrative from the project brief; this repo does not claim to reproduce them. CUDA performance paths require a compatible NVIDIA GPU and installed dependencies. CPU mode works without a GPU.

## Pipeline
1. Generate or load product text rows.
2. Batch-encode text into normalized embeddings (deterministic hashing encoder by default; optional Hugging Face/PyTorch encoder).
3. Compute chunked cosine similarities, excluding self-matches.
4. Emit candidate pairs with similarity threshold and top-k limit.
5. Write Parquet and run correctness/performance benchmarks.

## Quick start (CPU, no GPU required)
Python 3.10+ recommended.
```bash
python -m venv .venv
# activate environment
pip install -e '.[dev]'
python -m dedup_lab.cli generate --rows 10000 --out data/products.jsonl
python -m dedup_lab.cli run --input data/products.jsonl --output output/pairs.parquet --backend numpy --top-k 5 --threshold 0.88 --chunk-size 512
python -m dedup_lab.cli benchmark --rows 3000 --backend numpy
pytest -q
```
For environments without Parquet support, install `pyarrow` (included in the optional `dev` extra) or use `--output output/pairs.csv`.

## GPU setup
Install a PyTorch build matching your CUDA version first, then:
```bash
pip install -e '.[gpu]'
```
CuPy package version must match the installed CUDA runtime (the extra uses `cupy-cuda12x`; use a matching CuPy wheel for your system if needed). Triton is supported on Linux with compatible NVIDIA hardware. Run `python -m dedup_lab.cli doctor` to inspect optional capabilities.

```bash
python -m dedup_lab.cli run --input data/products.jsonl --output output/pairs.parquet --backend torch --device cuda --top-k 5
python -m dedup_lab.cli run --input data/products.jsonl --output output/pairs.parquet --backend cupy --device cuda --top-k 5
python -m dedup_lab.cli run --input data/products.jsonl --output output/pairs.parquet --backend triton --device cuda --top-k 5
```
`torch` uses chunked matrix multiplication. `cupy` uses chunked CuPy matrix multiplication. `triton` uses a custom row-wise normalization kernel and PyTorch chunked matrix multiplication; this is a teaching implementation, not a fused full top-k similarity kernel.

## Input format
JSONL, one object per line, e.g. `{"id":"p-1","title":"Blue cotton shirt size M"}`. The `title` field is required; `id` is optional and defaults to line number. The built-in hashing encoder is intentionally simple and intended for pipeline experiments. For real semantic similarity, use `--encoder sentence-transformers --model MODEL_NAME` after installing the optional `nlp` extra.

## Profiling
- PyTorch Profiler: `python scripts/profile_torch.py --rows 10000`
- Nsight Systems: `nsys profile -o reports/nsys python -m dedup_lab.cli run ... --backend torch --device cuda`
- Nsight Compute: profile the Triton normalization kernel via `ncu --set full python -m dedup_lab.cli run ... --backend triton --device cuda`
- Compare CPU and GPU with `python -m dedup_lab.cli benchmark --rows 20000 --backend numpy|torch`.

## Limitations / production considerations
- Pairwise similarity remains O(N²); chunking controls memory, not total work. A 12M × 12M exact comparison is generally not practical without candidate generation, approximate nearest-neighbor indexing (FAISS), sharding, or domain blocking.
- The default encoder is a deterministic token hashing baseline, not a transformer model.
- The default candidate output contains directed top-k matches; use `--undirected` to canonicalize and deduplicate pair IDs.
- GPU results depend on GPU model, driver, CUDA, PyTorch/CuPy/Triton versions, data, precision, and chunk size. No benchmark claims are made without actually running on target hardware.
